import os
from typing import Dict, Any, Optional
import google.generativeai as genai
from PIL import Image
import io
from pathlib import Path
import hashlib
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")


def _parse_key_file(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    keys: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#") or s == ".":
            continue
        keys.append(s)
    return keys


def _get_api_key(provided_key: Optional[str] = None) -> str:
    keys = _get_all_keys(provided_key)
    if keys:
        return keys[0]
    raise RuntimeError("Google Gemini API key not found. Set GOOGLE_API_KEY/GEMINI_API_KEY or add secrets/gemini_api_key.txt or env/gemini_keys.txt")


def _get_all_keys(primary: Optional[str] = None) -> list[str]:
    """Collect possible API keys in priority order. Supports env/env file/secrets list.
    Order: provided > env var > secrets/gemini_api_key.txt > env/gemini_keys.txt (one per line).
    """
    keys: list[str] = []
    if primary:
        keys.append(primary)
    # Env
    for k in (os.environ.get("GOOGLE_API_KEY"), os.environ.get("GEMINI_API_KEY")):
        if k and k not in keys:
            keys.append(k)
    # secrets file (single or multi-line) search across likely roots
    try:
        here = Path(__file__).resolve()
        candidates = [
            here.parents[1] / "secrets" / "gemini_api_key.txt",      # submission/secrets
            here.parents[2] / "secrets" / "gemini_api_key.txt",      # project root / secrets
            here.parents[1].parent / "secrets" / "gemini_api_key.txt" # submission parent / secrets
        ]
        for p in candidates:
            if p.exists():
                for k in _parse_key_file(p):
                    if k not in keys:
                        keys.append(k)
    except Exception:
        pass
    # env/gemini_keys.txt (multiple keys) search across likely roots
    try:
        here = Path(__file__).resolve()
        candidates = [
            here.parents[1] / "env" / "gemini_keys.txt",            # submission/env
            here.parents[2] / "env" / "gemini_keys.txt",            # project root env
            here.parents[1].parent / "env" / "gemini_keys.txt"      # submission parent env
        ]
        for p in candidates:
            if p.exists():
                for k in _parse_key_file(p):
                    if k not in keys:
                        keys.append(k)
    except Exception:
        pass
    return [k for k in keys if k]


def _pil_to_jpeg_bytes(img: Image.Image, max_side: int = 640, quality: int = 82) -> bytes:
    # Downscale to speed up upload/processing and compress efficiently
    im = img.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def _hash_image_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def analyze_plant_with_gemini(
    img: Image.Image,
    predicted_species: str,
    confidence: float,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout_sec: float = 4.0,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    max_side: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Calls Gemini with the input image and context and returns a structured analysis.
    """
    # Collect keys in priority order (provided/env/secrets/env file)
    candidate_keys = _get_all_keys(api_key)
    if not candidate_keys:
        raise RuntimeError("No Gemini API keys available. Add to env, secrets/gemini_api_key.txt, or env/gemini_keys.txt")

    # Defaults optimized for fast mode; can be overridden by caller
    generation_config = {
        "temperature": 0.6 if temperature is None else float(temperature),
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 320 if max_output_tokens is None else int(max_output_tokens),
    }
    preferred_model = model_name or DEFAULT_MODEL

    prompt = (
        "You are an expert agronomist and plant pathologist.\n"
        f"The vision model predicted species or candidates: '{predicted_species}' (confidence {confidence:.2f}).\n"
        "Guidance on species decision:\n"
        "- If model confidence is >= 0.60, assume the predicted species is correct unless there is STRONG visual evidence otherwise.\n"
        "- If you disagree, present a 'Possible species' with visual rationale; avoid accusatory language.\n"
        "Output REQUIREMENTS (very important): Provide a bilingual, structured answer with two top-level sections labeled EXACTLY as below so the app can toggle languages.\n"
        "For each language, include subsections: Crop, Problem, Detailed Home Remedies (with measurements and step-by-step), Chemical/Commercial (product + dosage per liter and interval), Preventive Measures, Climate & Season (temperature range, rainfall, sunlight), Suitable Regions, Nutritional Profile (key vitamins/minerals), Maintenance Tips (watering, pruning, spacing, soil, fertilizer).\n"
        "### English\n"
        "#### Crop Identified\n"
        "- <name>\n"
        "#### Problem Detected\n"
        "- Disease/Pest/Deficiency (or 'Healthy') with 1-2 lines of rationale from the image.\n"
        "#### Solution\n"
        "- Organic/Home Remedies: bullet points with exact quantities (e.g., 5 ml neem oil + 1 liter water; steps) and application interval.\n"
        "- Chemical/Commercial: specific product examples with dosage (grams/ml per liter) and spray interval.\n"
        "- Preventive Measures: 3-6 bullet points.\n"
        "#### Climate & Season\n"
        "- Optimal temperature range (°C), rainfall, sunlight, and growing season.\n"
        "#### Suitable Regions\n"
        "- Typical regions/countries where this crop grows well.\n"
        "#### Nutritional Profile\n"
        "- Key vitamins/minerals and health benefits (short).\n"
        "#### Maintenance Tips\n"
        "- Watering, pruning, spacing, soil, and fertilizer pointers.\n"
        "\n"
        "### Hindi\n"
        "#### पहचान की गई फसल\n"
        "- <नाम>\n"
        "#### पहचानी गई समस्या\n"
        "- रोग/कीट/पोषक कमी (या 'स्वस्थ'); छवि से 1-2 पंक्तियों का कारण।\n"
        "#### समाधान\n"
        "- घरेलू उपाय: सटीक मात्रा (जैसे 5 ml नीम तेल + 1 लीटर पानी; चरण) और लगाने का अंतराल।\n"
        "- रासायनिक/वाणिज्यिक: उत्पाद उदाहरण, मात्रा (ग्राम/मिली प्रति लीटर) और छिड़काव अंतराल।\n"
        "- बचाव के उपाय: 3-6 बिंदु।\n"
        "#### जलवायु और मौसम\n"
        "- अनुकूल तापमान सीमा (°C), वर्षा, धूप, और मौसम।\n"
        "#### उपयुक्त क्षेत्र\n"
        "- वे क्षेत्र/देश जहाँ यह फसल अच्छी तरह उगती है।\n"
        "#### पोषण प्रोफ़ाइल\n"
        "- मुख्य विटामिन/खनिज और लाभ (संक्षेप में)।\n"
        "#### रख-रखाव सुझाव\n"
        "- सिंचाई, छंटाई, दूरी, मिट्टी और उर्वरक के बिंदु।\n"
        "Keep outputs practical and concise."
    )

    image_bytes = _pil_to_jpeg_bytes(img, max_side=(max_side or 640), quality=82)
    img_hash = _hash_image_bytes(image_bytes)

    @lru_cache(maxsize=64)
    def _cached_call(hash_key: str, species: str, conf: float, tok: int, temp: float, side: int, mdl: str) -> str:
        # Try each key with the preferred model, then fallback to flash, enforcing timeout
        def _attempt_call(use_key: str, use_model: str):
            genai.configure(api_key=use_key)
            m = genai.GenerativeModel(use_model)
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(
                    m.generate_content,
                    [
                        {"mime_type": "image/jpeg", "data": image_bytes},
                        prompt,
                    ],
                    generation_config=generation_config,
                )
                return future

        try:
            # Iterate keys
            last_err = None
            for k in candidate_keys:
                # 1) Preferred model
                fut = _attempt_call(k, preferred_model)
                try:
                    r = fut.result(timeout=timeout_sec)
                    return r.text if hasattr(r, "text") else str(r)
                except FuturesTimeout:
                    # try next strategy
                    last_err = "timeout"
                except Exception as e:
                    msg = str(e)
                    last_err = msg
                    if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
                        # Try flash with same key
                        fut2 = _attempt_call(k, "gemini-1.5-flash")
                        try:
                            r2 = fut2.result(timeout=timeout_sec)
                            return (r2.text if hasattr(r2, "text") else str(r2)) + "\n\n_(Auto-fallback to flash model due to quota)_"
                        except Exception as e2:
                            last_err = str(e2)
                            continue
                    else:
                        # Unknown error; try next key
                        continue
            # If we reach here, all keys failed
            if last_err == "timeout":
                raise FuturesTimeout()
            raise RuntimeError(last_err or "All keys/models failed")
        except FuturesTimeout:
            # Quick fallback (fast, deterministic text)
            return (
                f"Species: {species} (model conf {conf:.2f}).\n"
                "Analysis: Response timed out; providing quick guidance.\n"
                "- Check leaves for spots, discoloration, pests.\n"
                "- Maintain even moisture; avoid waterlogging.\n"
                "- Apply balanced NPK based on soil test.\n"
                "- Use safe home remedies (soap spray for aphids, neem for mites).\n"
                "Climate & Season: grow in warm, frost-free period; ensure good sun & airflow.\n"
                "Nutrients: rich in vitamins/minerals varies by crop; consult local guides."
            )
        except Exception as e:
            msg = str(e)
            return (
                f"Species: {species} (model conf {conf:.2f}).\n"
                "Analysis: Service error; providing quick guidance.\n"
                f"Details: {msg}"
            )

    start = time.time()
    text = _cached_call(
        img_hash,
        predicted_species,
        float(confidence),
        int(generation_config["max_output_tokens"]),
        float(generation_config["temperature"]),
        int(max_side or 640),
        str(model_name or DEFAULT_MODEL),
    )
    elapsed = time.time() - start

    return {
        "predicted_species": predicted_species,
        "confidence": confidence,
        "gemini_output": text,
        "latency_sec": round(elapsed, 3),
    }
