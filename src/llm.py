import os
import re

from google import genai


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def clean_generated_output(text):
    """
    Clean common LaTeX formatting artifacts generated
    by the LLM so scientific values display correctly
    in normal Streamlit Markdown.
    """

    if not text:
        return text

    # ---------------------------------------------------------
    # Degree symbols
    # ---------------------------------------------------------

    text = re.sub(
        r"\\text\{\}\s*\\text\{\^\\circ\s*C\}",
        "°C",
        text
    )

    text = re.sub(
        r"\\text\{\^\\circ\s*C\}",
        "°C",
        text
    )

    text = re.sub(
        r"\\text\{\}\s*\\text\{\^\\circ\}",
        "°",
        text
    )

    text = re.sub(
        r"\\text\{\^\\circ\}",
        "°",
        text
    )

    text = text.replace(
        r"\circ",
        "°"
    )


    # ---------------------------------------------------------
    # Multiplication
    # ---------------------------------------------------------

    text = text.replace(
        r"\times",
        "×"
    )

    text = text.replace(
        r"\cdot",
        "·"
    )


    # ---------------------------------------------------------
    # Superscripts
    #
    # Convert common scientific notation such as:
    #
    # 10^{-3}
    # s^{-1}
    # x^{2}
    # ---------------------------------------------------------

    superscript_map = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
        "+": "⁺",
        "-": "⁻",
        "=": "⁼",
        "(": "⁽",
        ")": "⁾",
        "n": "ⁿ",
        "i": "ⁱ"
    }


    def convert_superscript(match):

        content = match.group(1)

        result = ""

        for character in content:

            result += superscript_map.get(
                character,
                character
            )

        return result


    text = re.sub(
        r"\^\{([^{}]+)\}",
        convert_superscript,
        text
    )


    # ---------------------------------------------------------
    # Simple ^ notation
    #
    # Example:
    #
    # x^2
    # 10^-3
    # ---------------------------------------------------------

    def convert_simple_superscript(match):

        content = match.group(1)

        result = ""

        for character in content:

            result += superscript_map.get(
                character,
                character
            )

        return result


    text = re.sub(
        r"\^([0-9+\-()])",
        convert_simple_superscript,
        text
    )


    # ---------------------------------------------------------
    # Remove common LaTeX text wrappers
    # ---------------------------------------------------------

    text = re.sub(
        r"\\text\{([^{}]*)\}",
        r"\1",
        text
    )

    text = text.replace(
        r"\text{}",
        ""
    )


    # ---------------------------------------------------------
    # LaTeX spacing commands
    # ---------------------------------------------------------

    text = text.replace(
        r"\,",
        " "
    )

    text = text.replace(
        r"\;",
        " "
    )

    text = text.replace(
        r"\:",
        " "
    )

    text = text.replace(
        r"\!",
        ""
    )


    # ---------------------------------------------------------
    # Remove display-math markers
    # ---------------------------------------------------------

    text = text.replace(
        "$$",
        ""
    )

    text = text.replace(
        "$",
        ""
    )


    # ---------------------------------------------------------
    # Clean accidental double spaces
    # ---------------------------------------------------------

    text = re.sub(
        r"[ \t]{2,}",
        " ",
        text
    )


    return text
    """
    Clean common LaTeX formatting artifacts generated
    by the LLM so that scientific values display
    correctly in Streamlit.
    """

    if not text:
        return text

    # ---------------------------------------------------------
    # Degree Celsius formatting
    # ---------------------------------------------------------

    text = re.sub(
        r"\\text\{\}\s*\\text\{\^\\circ\s*C\}",
        "°C",
        text
    )

    text = re.sub(
        r"\\text\{\^\\circ\s*C\}",
        "°C",
        text
    )

    text = re.sub(
        r"\\text\{\}\s*\\text\{\^\\circ\}",
        "°",
        text
    )

    text = re.sub(
        r"\\text\{\^\\circ\}",
        "°",
        text
    )

    # ---------------------------------------------------------
    # Other common LaTeX artifacts
    # ---------------------------------------------------------

    text = text.replace(
        r"\text{}",
        ""
    )

    text = text.replace(
        r"\text{",
        ""
    )

    text = text.replace(
        r"\circ",
        "°"
    )

    # ---------------------------------------------------------
    # Clean accidental LaTeX spacing commands
    # ---------------------------------------------------------

    text = text.replace(
        r"\,",
        " "
    )

    text = text.replace(
        r"\;",
        " "
    )

    text = text.replace(
        r"\!",
        ""
    )

    return text


def generate_answer(prompt):
    """
    Send a prompt to Gemini and return a cleaned
    generated answer.
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return clean_generated_output(
        response.text
    )