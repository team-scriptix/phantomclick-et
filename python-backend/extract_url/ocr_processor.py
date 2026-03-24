import base64
import io
import numpy as np
import cv2
from PIL import Image
import pytesseract
from typing import Dict, List, Tuple, Optional


class OCRProcessor:
    """
    Process SMS screenshots using OCR.

    Strategy:
      1. Detect UI chrome (status bar, nav bar, input box) and black it out
      2. Run tesseract on the masked image — single message or conversation,
         it doesn't matter; tesseract reads whatever text is present
      3. Build a word list with accurate char offsets for bbox lookup

    No bubble detection. No region splitting. Tesseract handles layout.
    """

    def __init__(self):
        # psm 6: assume uniform block of text — works well for SMS content
        self.tesseract_config = '--psm 6 --oem 3'

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                          #
    # ------------------------------------------------------------------ #
    def _preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """Boost contrast so low-saturation colored text (links) becomes readable"""
        cv_img = np.array(image)
        
        # Convert to LAB — boost L channel contrast
        lab = cv2.cvtColor(cv_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Also run a sharpening pass
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
    
        return Image.fromarray(enhanced)
    
    def process_image(self, image_data: bytes) -> Dict:
        """Extract text and word-level bounding boxes from image bytes."""
        try:
            pil_image = Image.open(io.BytesIO(image_data))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            cv_image = np.array(pil_image)

            # Mask out chrome so its tokens never pollute the word list
            content_mask = self._detect_chrome_mask(cv_image)
            masked_image = self._apply_mask(pil_image, content_mask)

            # Single OCR pass on the masked image
            result = self._ocr_image(masked_image)

            return {
                'extracted_text':   result['text'],
                'words':            result['words'],
                'image_dimensions': pil_image.size,
                'success':          True,
            }

        except Exception as e:
            print(f"OCR Processing Error: {e}")
            return {
                'extracted_text':   '',
                'words':            [],
                'image_dimensions': (0, 0),
                'success':          False,
                'error':            str(e),
            }

    def process_base64(self, base64_string: str) -> Dict:
        """Process a base64-encoded image string."""
        try:
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            return self.process_image(base64.b64decode(base64_string))
        except Exception as e:
            print(f"Base64 Processing Error: {e}")
            return {
                'extracted_text':   '',
                'words':            [],
                'image_dimensions': (0, 0),
                'success':          False,
                'error':            str(e),
            }

    def validate_image(self, image_data: bytes) -> Tuple[bool, Optional[str]]:
        """Basic sanity checks before processing."""
        try:
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size
            if width < 100 or height < 100:
                return False, "Image too small"
            if width > 4000 or height > 4000:
                return False, "Image too large"
            if image.format not in ['PNG', 'JPEG', 'JPG']:
                return False, "Invalid image format"
            return True, None
        except Exception as e:
            return False, f"Invalid image: {e}"

    # ------------------------------------------------------------------ #
    #  CHROME MASKING                                                      #
    # ------------------------------------------------------------------ #

    def _detect_chrome_mask(self, cv_image: np.ndarray) -> np.ndarray:
        """
        Return a binary mask: 255 = message content, 0 = UI chrome.

        Identifies chrome by scanning for horizontal bands of near-uniform
        colour — the universal signature of status bars and nav bars regardless
        of phone model, OS version, or dark/light mode.

        Scans top 20% for status bar / app header.
        Scans bottom 25% for nav bar / input field.
        """
        h, w   = cv_image.shape[:2]
        mask   = np.ones((h, w), dtype=np.uint8) * 255
        gray   = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        band_h = max(4, h // 60)   # ~1.7% of image height per scan band

        def mask_uniform_bands(y_start: int, y_end: int):
            for y in range(y_start, y_end, band_h):
                band = gray[y: y + band_h, :]
                if band.size and np.std(band) < 18:  # near-uniform = chrome
                    mask[y: y + band_h, :] = 0

        mask_uniform_bands(0,            int(h * 0.20))  # top chrome
        mask_uniform_bands(int(h * 0.75), h)             # bottom chrome

        # Also catch input fields: high edge density near bottom
        bottom = gray[int(h * 0.80):, :]
        edges  = cv2.Canny(bottom, 50, 150)
        strip_h = max(4, h // 40)
        for i in range(0, bottom.shape[0], strip_h):
            strip = edges[i: i + strip_h, :]
            if strip.sum() / (255 * max(strip.size, 1)) > 0.08:
                abs_y = int(h * 0.80) + i
                mask[abs_y: abs_y + strip_h, :] = 0

        return mask

    def _apply_mask(
        self,
        image: Image.Image,
        mask:  np.ndarray,
    ) -> Image.Image:
        """Black out masked (chrome) regions in the image."""
        cv_image          = np.array(image)
        masked            = cv_image.copy()
        masked[mask == 0] = 0
        return Image.fromarray(masked)

    # ------------------------------------------------------------------ #
    #  OCR                                                                 #
    # ------------------------------------------------------------------ #

    def _ocr_image(self, image: Image.Image) -> Dict:
        """
        Run tesseract on an image and return text + word list.

        Builds char_start / char_end offsets by counting only the tokens
        that pass _is_valid_token — the same tokens that go into the word
        list — so ScamBrain's bisect lookup always lands on the right word.
        """
        image = self._preprocess_for_ocr(image)  # add this line
        text = pytesseract.image_to_string(
            image, config=self.tesseract_config, lang='eng'
        ).strip()

        word_data = pytesseract.image_to_data(
            image, config=self.tesseract_config,
            lang='eng', output_type=pytesseract.Output.DICT,
        )

        words   = []
        running = 0   # char offset into the reconstructed token stream

        for i in range(len(word_data['text'])):
            token = word_data['text'][i].strip()
            if not token or not self._is_valid_token(token):
                continue

            conf = int(word_data['conf'][i])
            is_url_like = '.' in token and any(
                tld in token.lower() for tld in ['.in', '.com', '.net', '.org', '.co']
            )
            min_conf = 30 if is_url_like else 60

            if conf < min_conf:
                continue

            words.append({
                'text':       token,
                'char_start': running,
                'char_end':   running + len(token),
                'bbox': {
                    'x0': word_data['left'][i],
                    'y0': word_data['top'][i],
                    'x1': word_data['left'][i] + word_data['width'][i],
                    'y1': word_data['top'][i]  + word_data['height'][i],
                },
                'confidence': float(word_data['conf'][i]) / 100.0,
            })
            running += len(token) + 1

        return {'text': text, 'words': words}


    def _is_valid_token(self, token: str) -> bool:
        """
        Reject OCR garbage (UI icons, lone symbols, box-drawing characters).
        A valid token must have at least 2 alphanumeric characters.
        Passes real words, numbers, phone fragments, URLs.
        Drops things like '])', 'Ww', '©', '>' from icon renders.
        """
        return sum(1 for c in token if c.isalnum()) >= 2