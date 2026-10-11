import pymupdf
import re


def looks_like_reference_page(text):
    """
    Detect whether a PDF page belongs to a
    reference/bibliography section.

    The detector intentionally uses strict signals
    to avoid incorrectly classifying normal research
    pages as references.
    """

    if not text:
        return False

    text = text.strip()


    # ---------------------------------------------------------
    # 1. Detect an actual standalone reference heading
    # ---------------------------------------------------------

    reference_heading_pattern = (
        r"(?im)^\s*"
        r"(references|bibliography|literature cited|"
        r"references and notes|reference list)"
        r"\s*$"
    )

    if re.search(
        reference_heading_pattern,
        text
    ):
        return True


    # ---------------------------------------------------------
    # 2. Detect standalone numbered reference markers
    #
    # Example:
    #
    # [4]
    # Author...
    #
    # [5]
    # Author...
    #
    # [6]
    # Author...
    #
    # IMPORTANT:
    # We require the marker to occupy its own line.
    # This prevents normal citations such as
    # "according to [1]" from triggering the detector.
    # ---------------------------------------------------------

    standalone_markers = re.findall(
        r"(?m)^\s*\[\s*\d+\s*\]\s*$",
        text
    )


    # Three or more standalone markers strongly indicate
    # a bibliography/reference page.

    if len(standalone_markers) >= 3:
        return True


    return False


def extract_text_from_pdf(pdf_path):
    """
    Extract text from every page of a PDF.

    Reference pages are marked with:
        is_reference = True
    """

    document = pymupdf.open(pdf_path)

    pages = []

    reference_section_started = False


    for page_number, page in enumerate(document):

        text = page.get_text().strip()


        # -----------------------------------------------------
        # Check whether this page itself is a reference page
        # -----------------------------------------------------

        current_page_is_reference = (
            looks_like_reference_page(text)
        )


        # Once a genuine reference section has been detected,
        # subsequent pages are also considered reference pages.

        if current_page_is_reference:
            reference_section_started = True


        is_reference = reference_section_started


        pages.append({
            "page_number": page_number + 1,
            "text": text,
            "is_reference": is_reference
        })


        # Debug output

        if is_reference:

            print(
                f"[REFERENCE PAGE] "
                f"Page {page_number + 1}"
            )


    document.close()

    return pages