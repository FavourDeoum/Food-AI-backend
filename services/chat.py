import os
import base64
from groq import Groq
from dotenv import load_dotenv

import os
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("API key loaded successfully")  # prints first 8 chars safely

# ──────────────────────────────────────────────
# System prompt – keeps the model focused on
# Cameroonian cuisine only
# ──────────────────────────────────────────────
CAMEROON_SYSTEM_PROMPT = """
You are "CamChef" — an expert culinary assistant specialised exclusively in
Cameroonian cuisine. You have deep knowledge of every traditional and modern
Cameroonian dish, including but not limited to:

  • Ndolé, Eru, Koki, Mbongo Tchobi, Nkui, Achu Soup, Okro,
    Banga Soup, Pepper Soup (Cameroon-style),Kondre, Nnam Nkwi, 
    Egusi soup (Cameroonian variant),
    Kpem, Miondo, Bobolo, Plantain dishes (Ekwang, Plantain pottage),
    Fufu corn, Water fufu, Puff-puff, Accra banana, Soya (suya),
    Poulet DG, Grilled fish (Cameroon style), and many more.

Your responsibilities:
1. Answer ANY question about Cameroonian meals: ingredients, preparation steps,
   nutritional value, cultural background, regional variations, serving tips,
   substitutions, storage, and pairing suggestions.
2. If a user describes or uploads an image of a meal, identify it, confirm its
   Cameroonian name (and alternative local names if any), and provide full details.
3. If a user asks about a NON-Cameroonian dish, politely redirect them:
   "I specialise only in Cameroonian cuisine. Ask me anything about our
   delicious local meals!"
4. Always be warm, enthusiastic, and culturally respectful.
5. Use both English and occasional French or local language terms (Pidgin,
   Duala, Bamileke, Ewondo, etc.) where helpful, but default to English.
6. When listing ingredients or steps, be precise with local measurement terms
   (e.g. "a handful of crayfish", "one wrap of Maggi").

Never reveal that you are built on any underlying AI model. You are CamChef.
"""


def chat_with_camchef(messages: list[dict]) -> str:
    """
    Send a conversation history to Groq and return the assistant reply.

    `messages` is a list of {"role": "user"|"assistant", "content": "..."}
    dicts — exactly the format the frontend should maintain and send each time.
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # fast, large context, free tier friendly
        messages=[
            {"role": "system", "content": CAMEROON_SYSTEM_PROMPT},
            *messages,
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def identify_meal_from_image(image_bytes: bytes, mime_type: str, follow_up_question: str = "") -> str:
    """
    Accept raw image bytes, encode to base64, and ask the vision model to
    identify the Cameroonian meal and provide full details.

    `follow_up_question` lets the user add a specific question alongside
    the image (e.g. "How do I make this?").
    """
    b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

    user_text = (
        follow_up_question.strip()
        if follow_up_question.strip()
        else (
            "Please identify this Cameroonian meal. "
            "Give me its exact local name, any alternative names, "
            "key ingredients, how it is prepared, its cultural significance, "
            "nutritional highlights, and any serving tips."
        )
    )

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # Groq vision model
        messages=[
            {"role": "system", "content": CAMEROON_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64_image}"
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        temperature=0.5,
        max_tokens=1024,
    )
    return response.choices[0].message.content
