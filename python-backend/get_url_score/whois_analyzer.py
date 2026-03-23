import re
import socket
import sys
import os
from datetime import datetime

import requests
import whois


class WHOISAnalyzer:
    """
    Domain age/registration lookup with a 3-tier fallback chain:
      1. python-whois  (free, port 43)
      2. RDAP          (free, HTTPS)
      3. WhoisFreaks   (paid API, only if api_key provided)
    """

    TLD_RDAP = {
        "in":  "https://rdap.registry.in/domain/{}",
        "tk":  "https://rdap.dot.tk/domain/{}",
        "com": "https://rdap.verisign.com/com/v1/domain/{}",
        "net": "https://rdap.verisign.com/net/v1/domain/{}",
        "org": "https://rdap.publicinterestregistry.org/rdap/domain/{}",
    }

    RDAP_FALLBACKS = [
        "https://rdap.org/domain/{}",
        "https://rdap.iana.org/domain/{}",
        "https://www.rdap.net/domain/{}",
    ]

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def analyze_domain(self, domain: str) -> dict:
        """Return registration metadata for *domain*."""

        # IP addresses have no WHOIS record
        if re.match(r'^(\d{1,3}\.){3}\d{1,3}\.?$', domain):
            print(f"⏭️  Skipping WHOIS for IP: {domain}")
            return self._ip_result()

        # Try each method in order; stop at first success
        for method in (
            self._try_python_whois,
            self._try_rdap,
            self._try_whoisfreaks,   # no-op when api_key is None
        ):
            try:
                result = method(domain)
                if result:
                    return result
            except Exception as e:
                print(f"⚠️  {method.__name__} failed for {domain}: {e}")

        print(f"❌ All WHOIS methods failed for {domain}")
        return self._default_result()

    # ------------------------------------------------------------------ #
    #  Method 1 — python-whois                                            #
    # ------------------------------------------------------------------ #

    def _try_python_whois(self, domain: str) -> dict | None:
        if not self._port43_open():
            print("⚠️  Port 43 closed — skipping python-whois")
            return None

        devnull = open(os.devnull, "w")
        old_stderr, sys.stderr = sys.stderr, devnull
        try:
            w = whois.whois(domain)
        finally:
            sys.stderr = old_stderr
            devnull.close()

        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if not creation:
            return None

        creation = self._strip_tz(creation)
        age_days = (datetime.now() - creation).days
        print(f"✅ python-whois OK: {creation.date()}")
        return self._build_result(
            age_days=age_days,
            creation=creation,
            registrar=getattr(w, "registrar", None) or "Unknown",
        )

    # ------------------------------------------------------------------ #
    #  Method 2 — RDAP                                                    #
    # ------------------------------------------------------------------ #

    def _try_rdap(self, domain: str) -> dict | None:
        tld = domain.rsplit(".", 1)[-1].lower()

        endpoints = []
        if tld in self.TLD_RDAP:
            endpoints.append(self.TLD_RDAP[tld].format(domain))
        endpoints += [url.format(domain) for url in self.RDAP_FALLBACKS]

        data = None
        for url in endpoints:
            try:
                print(f"🌐 RDAP: {url}")
                r = requests.get(
                    url,
                    timeout=15,
                    headers={"User-Agent": "phishing-detector/1.0"},
                )
                if r.status_code == 200:
                    data = r.json()
                    break
                print(f"   → HTTP {r.status_code}")
            except requests.exceptions.Timeout:
                print(f"   → timeout")
            except requests.exceptions.ConnectionError:
                print(f"   → connection error")

        if not data:
            return None

        creation = self._parse_rdap_date(data)
        if not creation:
            return None

        age_days = (datetime.now() - creation).days
        registrar = self._parse_rdap_registrar(data)
        print(f"✅ RDAP OK: {creation.date()}")
        return self._build_result(age_days=age_days, creation=creation, registrar=registrar)

    def _parse_rdap_date(self, data: dict) -> datetime | None:
        for event in data.get("events", []):
            if event.get("eventAction") == "registration":
                raw = event.get("eventDate", "")
                try:
                    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                    return self._strip_tz(dt)
                except ValueError:
                    pass
        return None

    def _parse_rdap_registrar(self, data: dict) -> str:
        for entity in data.get("entities", []):
            if "registrar" not in entity.get("roles", []):
                continue
            try:
                vcard = entity.get("vcardArray", [None, []])[1]
                for item in (vcard or []):
                    if isinstance(item, list) and item[0] == "org":
                        name = item[3]
                        return name if isinstance(name, str) else "Unknown"
            except (IndexError, TypeError):
                pass
        return "Unknown"

    # ------------------------------------------------------------------ #
    #  Method 3 — WhoisFreaks (paid, optional)                           #
    # ------------------------------------------------------------------ #

    def _try_whoisfreaks(self, domain: str) -> dict | None:
        if not self.api_key:
            return None

        r = requests.get(
            "https://api.whoisfreaks.com/v1.0/whois",
            params={"apiKey": self.api_key, "whois": "live", "domainName": domain},
            timeout=10,
        )
        if r.status_code != 200:
            raise RuntimeError(f"WhoisFreaks HTTP {r.status_code}")

        payload = r.json()
        raw_date = (
            payload.get("create_date")
            or (payload.get("domain_registrar") or {}).get("registration_date")
        )
        if not raw_date:
            raise RuntimeError("WhoisFreaks: no creation date in response")

        creation = datetime.strptime(raw_date[:10], "%Y-%m-%d")
        age_days = (datetime.now() - creation).days
        registrar = (payload.get("domain_registrar") or {}).get("registrar_name", "Unknown")
        print(f"✅ WhoisFreaks OK: {creation.date()}")
        return self._build_result(age_days=age_days, creation=creation, registrar=registrar)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _port43_open() -> bool:
        try:
            s = socket.create_connection(("whois.iana.org", 43), timeout=2)
            s.close()
            return True
        except OSError:
            return False

    @staticmethod
    def _strip_tz(dt: datetime) -> datetime:
        return dt.replace(tzinfo=None) if dt.tzinfo else dt

    @staticmethod
    def _build_result(age_days: int, creation: datetime, registrar: str) -> dict:
        return {
            "domain_age_days": age_days,
            "creation_date": creation.isoformat(),
            "is_recently_created": int(age_days < 30),
            "is_very_new": int(age_days < 3),
            "registrar": registrar,
            "status": "success",
        }

    @staticmethod
    def _default_result() -> dict:
        return {
            "domain_age_days": -1,
            "creation_date": None,
            "is_recently_created": 0,
            "is_very_new": 0,
            "registrar": "Unknown",
            "status": "failed",
        }

    @staticmethod
    def _ip_result() -> dict:
        return {
            "domain_age_days": -1,
            "creation_date": None,
            "is_recently_created": 0,
            "is_very_new": 0,
            "registrar": "Unknown",
            "status": "ip_address",
        }