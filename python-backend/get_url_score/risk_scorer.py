import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import tldextract
import Levenshtein
import re

try:
    from .feature_extractor import URLFeatureExtractor
    from .whois_analyzer import WHOISAnalyzer
except ImportError:
    from feature_extractor import URLFeatureExtractor
    from whois_analyzer import WHOISAnalyzer


# ---------------------------------------------------------------------------
# Known-safe domains — exact registrable domain match only
# ---------------------------------------------------------------------------
KNOWN_SAFE = {
    # Banks
    "sbi.co.in", "hdfcbank.com", "icicibank.com", "axisbank.com",
    "bankofindia.co.in", "bankofindia.in", "bankofbaroda.in",
    "pnbindia.in", "canarabank.com", "unionbankofindia.co.in",
    "kotakbank.com", "yesbank.in",
    # Payments
    "paytm.com", "phonepe.com", "gpay.app", "bhimupi.org.in", "razorpay.com",
    # Shopping
    "myntra.com", "flipkart.com", "amazon.in", "amazon.com",
    "meesho.com", "snapdeal.com",
    # Tech
    "google.com", "microsoft.com", "apple.com",
    "youtube.com", "linkedin.com", "github.com",
    # Govt
    "incometax.gov.in", "irctc.co.in", "uidai.gov.in",
    "npci.org.in", "rbi.org.in",
}

# Weights — lookalike and ML carry the most signal
WEIGHTS = {
    "lookalike":   0.35,
    "domain_age":  0.15,
    "entropy":     0.08,
    "pattern":     0.17,
    "ml":          0.25,
}

# Brand keywords — sorted longest-first so prefix matching is greedy
BRAND_KEYWORDS = sorted({
    # Explicit bank names
    "sbi", "hdfc", "icici", "axis", "kotak", "pnb", "canara",
    "union", "baroda", "yesbank", "iob", "boi",
    # Generic banking terms phishers love
    "bank", "banking", "netbanking",
    # Payments
    "paytm", "phonepe", "googlepay", "bhim", "upi", "razorpay",
    # Govt
    "irctc", "incometax", "aadhar", "uidai",
}, key=len, reverse=True)


def _detect_brand(domain: str, url: str) -> str | None:
    """
    Returns the matched brand keyword or None.
    Splits domain+URL on hyphens and dots, then checks each token:
      - exact match in BRAND_KEYWORDS
      - OR token starts with a brand keyword (catches sbionline, hdfcnetbanking etc.)
    Longest keywords are checked first to avoid short false matches.
    """
    tokens = re.split(r"[-.]", (domain + " " + url).lower())
    for token in tokens:
        if not token:
            continue
        for kw in BRAND_KEYWORDS:
            if len(kw) < 3:
                continue
            if token == kw or token.startswith(kw):
                return kw
    return None


class RiskScorer:

    def __init__(self):
        BASE_DIR  = Path(__file__).resolve().parent
        model_dir = BASE_DIR / "models"

        self.model         = joblib.load(model_dir / "phishing_detector_rf.pkl")
        self.feature_names = joblib.load(model_dir / "feature_names.pkl")

        self.url_extractor  = URLFeatureExtractor()
        self.whois_analyzer = WHOISAnalyzer(api_key="2441ba19e4c2467891be4c83f3858c71")

        self.legit_domains = {
            "sbi.co.in", "google.com", "paypal.com",
            "amazon.in", "amazon.com", "microsoft.com",
        }

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate_risk_score(self, url: str) -> dict:

        extracted = tldextract.extract(url)
        domain    = f"{extracted.domain}.{extracted.suffix}"

        # Fast-path: known safe
        if domain in KNOWN_SAFE:
            return {
                "url": url,
                "analysis": {"domain": domain},
                "risk_breakdown": {},
                "overall_threat_score": 5,
                "risk_level": "Minimal",
                "trigger_ghostview": False,
                "ml_prediction_score": 0,
            }

        result = {
            "url": url,
            "analysis": {},
            "risk_breakdown": {},
            "overall_threat_score": 0,
            "risk_level": "Unknown",
            "trigger_ghostview": False,
            "ml_prediction_score": 0,
        }

        try:
            url_features = self.url_extractor.extract_all_features(url)

            # Override feature_extractor's narrow brand check with our richer one
            matched_brand = _detect_brand(domain, url)
            url_features["has_brand_name"]   = 1 if matched_brand else 0
            url_features["matched_brand"]     = matched_brand or ""

            whois_data = self.whois_analyzer.analyze_domain(domain)

            result["analysis"] = {
                "domain":       domain,
                "url_features": url_features,
                "whois":        whois_data,
            }

            # ---- individual components (all 0–100) ----
            lookalike_risk  = self._calc_lookalike(domain, url_features)
            domain_age_risk = self._calc_domain_age(whois_data, url_features)
            entropy_risk    = self._calc_entropy(url_features)
            pattern_risk    = self._calc_pattern(url_features)
            ml_score        = self._ml_score(url_features)

            breakdown = {
                "lookalike":  lookalike_risk,
                "domain_age": domain_age_risk,
                "entropy":    entropy_risk,
                "pattern":    pattern_risk,
                "ml":         int(ml_score),
            }
            result["risk_breakdown"] = breakdown

            # ---- weighted sum ----
            score = sum(breakdown[k] * WEIGHTS[k] for k in WEIGHTS)

            # ---- hard overrides (applied in ascending order of severity) ----
            lev       = url_features.get("levenshtein_distance", 999)
            sim       = url_features.get("lookalike_score", 0)
            has_brand = url_features.get("has_brand_name", 0) == 1
            sus_kw    = url_features.get("suspicious_keyword_count", 0)
            age_days  = whois_data.get("domain_age_days", -1)
            ghost     = whois_data.get("status") == "failed"

            # IP-based URL
            if url_features.get("has_ip", 0) == 1:
                score = max(score, 65)

            # ML very confident
            if ml_score >= 90:
                score = max(score, 80)

            # Clear typosquat
            if lev <= 3 and sim >= 80:
                score = max(score, 75)

            # Typosquat + ghost/new domain
            if lev <= 3 and sim >= 80 and age_days < 30:
                score = max(score, 85)

            # Ghost domain + brand
            if ghost and has_brand:
                score = max(score, 75)

            # Ghost domain + brand + login/kyc keywords
            if ghost and has_brand and sus_kw >= 2:
                score = max(score, 85)

            score = min(int(float(score)), 100)

            result["overall_threat_score"] = score
            result["ml_prediction_score"]  = int(float(ml_score))
            result["risk_level"]           = self._risk_level(score)
            result["trigger_ghostview"]    = score >= 20

        except Exception as e:
            result["error"] = str(e)

        return result

    # ------------------------------------------------------------------ #
    #  Risk components                                                     #
    # ------------------------------------------------------------------ #

    def _calc_lookalike(self, domain: str, features: dict) -> int:
        if domain in self.legit_domains:
            return 0
        if features.get("levenshtein_distance", -1) == 0:
            return 0

        lev       = features.get("levenshtein_distance", 999)
        sim       = features.get("lookalike_score", 0)
        closest   = features.get("closest_legit_domain", "")
        has_brand = features.get("has_brand_name", 0) == 1

        # Tight typosquat
        if lev <= 2 and sim >= 85:
            return min(int(sim), 100)

        # Looser typosquat
        if lev <= 3 and sim > 70:
            return min(int(sim), 95)

        # Brand keyword in domain but not the real domain
        if has_brand:
            return max(70, min(int(sim), 95)) if sim > 60 else 65

        # Elongated lookalike (baaaannkkooffindia.com)
        if self._is_elongated_lookalike(domain, closest):
            return 85

        return 0

    def _is_elongated_lookalike(self, domain: str, closest_legit: str) -> bool:
        if not closest_legit:
            return False
        collapse    = lambda s: re.sub(r"(.)\1+", r"\1", s.split(".")[0].lower())
        d, l        = collapse(domain), collapse(closest_legit)
        dist        = Levenshtein.distance(d, l)
        max_len     = max(len(d), len(l))
        return ((1 - dist / max_len) * 100 if max_len else 0) > 75

    def _calc_domain_age(self, whois_data: dict, features: dict) -> int:
        age       = whois_data.get("domain_age_days")
        has_brand = features.get("has_brand_name", 0) == 1
        sus_kw    = features.get("suspicious_keyword_count", 0)
        sus_tld   = features.get("has_suspicious_tld", 0) == 1
        lev       = features.get("levenshtein_distance", 999)

        if age is None or age < 0:
            if has_brand and sus_kw >= 2: return 90
            if has_brand and lev <= 3:    return 85
            if has_brand or sus_kw >= 2:  return 65
            if sus_tld:                   return 60
            return 25

        if age < 3:    return 100
        if age < 30:   return 75
        if age < 180:  return 50
        if age < 365:  return 30
        return 10

    def _calc_entropy(self, features: dict) -> int:
        e = features.get("entropy", 0)
        if e > 4.5: return 80
        if e > 4.0: return 60
        if e > 3.5: return 40
        return 20

    def _calc_pattern(self, features: dict) -> int:
        risk  = min(features.get("suspicious_keyword_count", 0) * 15, 45)
        risk += 40 if features.get("has_ip", 0) == 1 else 0
        risk += 25 if features.get("has_suspicious_tld", 0) == 1 else 0
        risk += 20 if features.get("has_https", 1) == 0 else 0
        # No HTTPS on a domain impersonating a brand is extra suspicious
        if features.get("has_https", 1) == 0 and features.get("has_brand_name", 0) == 1:
            risk += 15
        return min(risk, 100)

    def _ml_score(self, features: dict) -> float:
        X = pd.DataFrame(
            [[features.get(f, 0) for f in self.feature_names]],
            columns=self.feature_names,
        )
        return self.model.predict_proba(X)[0][1] * 100

    @staticmethod
    def _risk_level(score: int) -> str:
        if score >= 80: return "Critical"
        if score >= 60: return "High"
        if score >= 40: return "Medium"
        if score >= 20: return "Low"
        return "Minimal"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scorer = RiskScorer()

    tests = [
        ("https://www.sbi.co.in",             "Minimal"),
        ("https://www.irctc.co.in",            "Minimal"),
        ("https://bankofindia.in",             "Minimal"),
        ("www.bankkofindiaa.co.in",            "Critical"),  # typosquat
        ("https://sbi-kyc-update.in",          "Critical"),  # brand + keywords
        ("https://hdfc-update-kyc.in",         "Critical"),  # brand + keywords
        ("https://secure-hdfc-netbanking.tk",  "Critical"),  # sus TLD
        ("https://icici-bankk-login.com",      "Critical"),  # ghost + brand
        ("baaaannkkkkoofffinddiaaa.com",       "Critical"),  # elongated
        ("https://sbionline-kyc.com",          "Critical"),  # prefix brand + keyword
        ("https://hdfcnetbanking-secure.com",  "Critical"),  # prefix brand
    ]

    print(f"\n{'URL':<48} {'EXP':<10} {'GOT':<10} {'SCORE'}")
    print("─" * 80)
    for url, expected in tests:
        r    = scorer.calculate_risk_score(url)
        flag = "✅" if r["risk_level"] == expected else "❌"
        print(f"{flag} {url:<47} {expected:<10} {r['risk_level']:<10} {r['overall_threat_score']}")
        print(f"   breakdown={r['risk_breakdown']}")