import streamlit as st
import re

from pathlib import Path



from src.ingestion import (

    ingest_pdf,

    calculate_document_id

)



from src.vector_store import (

    create_collection,

    delete_document

)



from src.rag import (

    answer_question,

    summarize_paper,

    extract_experimental_methods,

    extract_materials_and_alloys,

    extract_key_findings,

    detect_research_gaps,

    compare_papers

)


def remove_inline_page_citations(text):
    """Remove inline page citations while preserving technical numbers."""
    if not isinstance(text, str):
        return text

    text = re.sub(
        r"\s*\(\s*pp?\.?\s*\d+(?:\s*[-–—]\s*\d+)?"
        r"(?:\s*,\s*\d+(?:\s*[-–—]\s*\d+)?)*\s*\)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*\(\s*Page\s+\d+(?:\s*[-–—]\s*\d+)?"
        r"(?:\s*,\s*Page\s+\d+(?:\s*[-–—]\s*\d+)?)*\s*\)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text




# ============================================================

# PAGE CONFIGURATION

# ============================================================



st.set_page_config(

    page_title="Materials Research Assistant",

    page_icon="",

    layout="wide"

)




# ============================================================

# PATHS

# ============================================================



DOCUMENTS_DIR = Path("data/documents")



DOCUMENTS_DIR.mkdir(

    parents=True,

    exist_ok=True

)





# ============================================================

# CHROMADB COLLECTION

# ============================================================



collection = create_collection()





# ============================================================

# TITLE

# ============================================================



st.title(" Materials Research Assistant")



st.write(

    "An AI-powered assistant for exploring research papers "
    

)





# ============================================================

# UPLOAD RESEARCH PAPERS

# ============================================================



st.header(" Upload Research Papers")



uploaded_files = st.file_uploader(

    "Upload one or more PDF research papers",

    type=["pdf"],

    accept_multiple_files=True

)





if uploaded_files:



    for uploaded_file in uploaded_files:



        file_path = (

            DOCUMENTS_DIR

            / uploaded_file.name

        )



        # ----------------------------------------------------

        # Save uploaded PDF

        # ----------------------------------------------------



        with open(

            file_path,

            "wb"

        ) as file:



            file.write(

                uploaded_file.getbuffer()

            )





        # ----------------------------------------------------

        # Process and index PDF

        # ----------------------------------------------------



        result = ingest_pdf(

            str(file_path)

        )





        if result["status"] == "processed":



            st.success(

                f"Indexed: {uploaded_file.name} "

                f"({result['chunks']} chunks)"

            )





        elif result["status"] == "already_exists":



            st.info(

                f"Already indexed: "

                f"{uploaded_file.name}"

            )





# ============================================================

# INDEXED DOCUMENTS

# ============================================================



st.header(" Indexed Documents")





pdf_files = sorted(

    DOCUMENTS_DIR.glob("*.pdf")

)





if pdf_files:



    for pdf_file in pdf_files:



        document_id = calculate_document_id(

            pdf_file

        )





        col1, col2 = st.columns(

            [6, 1]

        )





        # ----------------------------------------------------

        # Document name

        # ----------------------------------------------------



        with col1:



            st.write(

                f"📄 **{pdf_file.name}**"

            )





        # ----------------------------------------------------

        # Delete button

        # ----------------------------------------------------



        with col2:



            if st.button(

                "🗑 Delete",

                key=f"delete_{pdf_file.name}"

            ):



                delete_document(

                    collection,

                    document_id

                )





                if pdf_file.exists():



                    pdf_file.unlink()





                st.success(

                    f"Deleted: {pdf_file.name}"

                )



                st.rerun()





else:



    st.info(

        "No research papers have been indexed yet."

    )





# ============================================================

# RESEARCH TOOLS

# ============================================================



st.header(" Research Tools")



st.write(

    "Choose a paper and select the type of research "

    "analysis you want to perform."

)





if pdf_files:



    # --------------------------------------------------------

    # Paper selector

    # --------------------------------------------------------



    paper_names = [

        pdf_file.name

        for pdf_file in pdf_files

    ]





    selected_paper = st.selectbox(

        " Select a research paper",

        paper_names

    )





    # --------------------------------------------------------

    # Analysis selector

    # --------------------------------------------------------



    analysis_type = st.selectbox(

        " Choose an analysis",

        [

            "Ask a Research Question",

            "Summarize Paper",

            "Extract Experimental Methods",

            "Extract Materials & Alloys",

            "Extract Key Findings",

            "Find Research Gaps",

            "Compare Research Papers"

        ]

    )





    # ========================================================

    # SUMMARIZE PAPER

    # ========================================================



    if analysis_type == "Summarize Paper":



        st.write(

            f"📄 Selected paper: "

            f"**{selected_paper}**"

        )





        if st.button(

            " Generate Paper Summary",

            key="generate_summary"

        ):



            with st.spinner(

                "Analyzing the research paper..."

            ):



                summary, summary_results = (

                    summarize_paper(

                        collection,

                        selected_paper

                    )

                )





            # ------------------------------------------------

            # Display summary

            # ------------------------------------------------



            st.header(

                " Research Paper Summary"

            )



            st.markdown(

                remove_inline_page_citations(summary)

            )





            # ------------------------------------------------

            # Display summary sources

            # ------------------------------------------------



            # Display summary sources grouped by document
            summary_metadatas = summary_results.get("metadatas", [[]])[0] or []
            summary_documents = summary_results.get("documents", [[]])[0] or []

            grouped_summary_sources = {}

            for i, metadata in enumerate(summary_metadatas):
                metadata = metadata or {}

                source = metadata.get("source", "Unknown source")
                page = metadata.get("page_number")

                if source not in grouped_summary_sources:
                    grouped_summary_sources[source] = set()

                if page is not None:
                    grouped_summary_sources[source].add(page)

            if grouped_summary_sources:
                st.header("Summary Sources")

                for source, pages in grouped_summary_sources.items():
                    with st.container(border=True):
                        st.markdown(f"**{source}**")

                        sorted_pages = sorted(
                            pages,
                            key=lambda p: (
                                not isinstance(p, (int, float)),
                                p if isinstance(p, (int, float)) else str(p),
                            ),
                        )

                        if sorted_pages:
                            page_text = ", ".join(
                                str(page) for page in sorted_pages
                            )
                            st.caption(f"Relevant pages: {page_text}")
                        else:
                            st.caption("Page numbers unavailable")





                    # ----------------------------------------

                    # Create source preview

                    # ----------------------------------------




    elif analysis_type == "Ask a Research Question":

        st.subheader("Ask a Research Question")

        # Form submission allows Enter to submit the question.
        with st.form(key="research_question_form"):

            question = st.text_input(
                "Enter your question:",
                placeholder="Ask something about your uploaded research papers..."
            )

            submitted = st.form_submit_button(
                "Ask Assistant",
                type="primary"
            )

        if submitted:

            if question.strip():

                with st.spinner("Searching the research papers..."):

                    answer, results = answer_question(
                        collection,
                        question.strip(),
                        top_k=5
                    )

                # Display answer
                st.header("Answer")
                st.markdown(answer)

                # Group sources by paper
                grouped_sources = {}

                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]

                for metadata in metadatas:

                    source = metadata.get("source", "Unknown paper")
                    page = metadata.get("page_number", "Unknown")

                    if source not in grouped_sources:
                        grouped_sources[source] = []

                    if page not in grouped_sources[source]:
                        grouped_sources[source].append(page)

                # Display compact source list
                if grouped_sources:

                    st.header("Sources")

                    for source, pages in grouped_sources.items():

                        with st.container(border=True):

                            st.markdown(f"**{source}**")

                            sorted_pages = sorted(
                                set(pages),
                                key=lambda p: (
                                    not isinstance(p, (int, float)),
                                    p if isinstance(p, (int, float)) else str(p)
                                )
                            )

                            page_text = ", ".join(
                                str(page) for page in sorted_pages
                            )

                            st.caption(f"Relevant pages: {page_text}")

                else:

                    st.info("No relevant sources were retrieved.")

            else:

                st.warning("Please enter a question.")




    # ========================================================

    # OTHER RESEARCH TOOLS

    # ========================================================



    elif analysis_type == "Extract Experimental Methods":



        st.write(

        f"📄 Selected paper: "

        f"**{selected_paper}**"

    )



        if st.button(

            " Extract Experimental Methods",

            key="extract_methods"

        ):



            with st.spinner(

                "Extracting experimental methods..."

            ):



                methods, method_results = (

                    extract_experimental_methods(

                        collection,

                        selected_paper

                    )

                )



            # ----------------------------------------------------

            # Display extracted methods

            # ----------------------------------------------------



            st.header(

                " Experimental Methods"

            )



            st.markdown(

                methods

            )



            # ----------------------------------------------------

            # Display evidence

            # ----------------------------------------------------



            method_documents = (

                method_results[

                    "documents"

                ][0]

            )



            if method_documents:



             
                # Display grouped evidence sources
                method_metadata = method_results.get("metadatas", [[]])[0]
                grouped_method_sources = {}

            for metadata in method_metadata:
                metadata = metadata or {}
                source = metadata.get("source", "Unknown source")
                page = metadata.get("page_number")

                if source not in grouped_method_sources:
                    grouped_method_sources[source] = set()

                if page is not None:
                    grouped_method_sources[source].add(page)

            if grouped_method_sources:
                st.header("Sources")

                for source, pages in grouped_method_sources.items():
                    with st.container(border=True):
                        st.markdown(f"**{source}**")

                        if pages:
                            sorted_pages = sorted(
                                pages,
                                key=lambda p: (
                                    not isinstance(p, (int, float)),
                                    p if isinstance(p, (int, float)) else str(p)
                                )
                            )
                            page_text = ", ".join(
                                str(page) for page in sorted_pages
                            )
                            st.caption(f"Relevant pages: {page_text}")
                        else:
                            st.caption("Page numbers unavailable")



            else:



                st.info(

                    "No experimental-method evidence "

                    "was retrieved."

                )





    elif analysis_type == "Extract Materials & Alloys":



        st.write(

            f"📄 Selected paper: "

            f"**{selected_paper}**"

        )



        if st.button(

            " Extract Materials & Alloys",

            key="extract_materials"

        ):



            with st.spinner(

                "Extracting materials and alloy information..."

            ):



                materials, material_results = (

                    extract_materials_and_alloys(

                        collection,

                        selected_paper

                    )

                )



            # ----------------------------------------------------

            # Display extracted information

            # ----------------------------------------------------



            st.header(

                " Materials & Alloy Analysis"

            )



            st.markdown(

                materials

            )



            # ----------------------------------------------------

            # Display evidence

            # ----------------------------------------------------



            material_documents = (

                material_results[

                    "documents"

                ][0]

            )



            if material_documents:



                
            # Display grouped evidence sources
             material_metadata = material_results.get("metadatas", [[]])[0]
            grouped_material_sources = {}

            for metadata in material_metadata:
                metadata = metadata or {}
                source = metadata.get("source", "Unknown source")
                page = metadata.get("page_number")

                if source not in grouped_material_sources:
                    grouped_material_sources[source] = set()

                if page is not None:
                    grouped_material_sources[source].add(page)

            if grouped_material_sources:
                st.header("Sources")

                for source, pages in grouped_material_sources.items():
                    with st.container(border=True):
                        st.markdown(f"**{source}**")

                        if pages:
                            sorted_pages = sorted(
                                pages,
                                key=lambda p: (
                                    not isinstance(p, (int, float)),
                                    p if isinstance(p, (int, float)) else str(p)
                                )
                            )
                            page_text = ", ".join(
                                str(page) for page in sorted_pages
                            )
                            st.caption(f"Relevant pages: {page_text}")
                        else:
                            st.caption("Page numbers unavailable")



            else:



                st.info(

                    "No material-related evidence "

                    "was retrieved."

                )





    elif analysis_type == "Extract Key Findings":



        st.write(

            f"📄 Selected paper: "

            f"**{selected_paper}**"

        )



        if st.button(

            " Extract Key Findings",

            key="extract_key_findings"

        ):



            with st.spinner(

                "Analyzing the key findings..."

            ):



                findings, finding_results = (

                    extract_key_findings(

                        collection,

                        selected_paper

                    )

                )



            # ----------------------------------------------------

            # Display findings

            # ----------------------------------------------------



            st.header(

                " Key Findings"

            )



            st.markdown(

                findings

            )



            # ----------------------------------------------------

            # Display evidence

            # ----------------------------------------------------



            finding_documents = (

                finding_results[

                    "documents"

                ][0]

            )



            if finding_documents:



                
                # Group finding evidence by document
                finding_metadata = finding_results.get("metadatas", [[]])[0]
                grouped_finding_sources = {}

                for metadata in finding_metadata:
                    metadata = metadata or {}
                    source = metadata.get("source", "Unknown source")
                    page = metadata.get("page_number")

                    if source not in grouped_finding_sources:
                        grouped_finding_sources[source] = set()

                    if page is not None:
                        grouped_finding_sources[source].add(page)

                if grouped_finding_sources:
                    st.header("Sources")

                    for source, pages in grouped_finding_sources.items():
                        with st.container(border=True):
                            st.markdown(f"**{source}**")

                            if pages:
                                sorted_pages = sorted(
                                    pages,
                                    key=lambda p: (
                                        not isinstance(p, (int, float)),
                                        p if isinstance(p, (int, float)) else str(p)
                                    )
                                )
                                page_text = ", ".join(
                                    str(page) for page in sorted_pages
                                )
                                st.caption(f"Relevant pages: {page_text}")
                            else:
                                st.caption("Page numbers unavailable")




            else:



                st.info(

                    "No finding-related evidence "

                    "was retrieved."

                )





    elif analysis_type == "Find Research Gaps":



        st.write(

            f"📄 Selected paper: "

            f"**{selected_paper}**"

        )



        if st.button(

            " Detect Research Gaps",

            key="detect_research_gaps"

        ):



            with st.spinner(

                "Analyzing research gaps..."

            ):



                gaps, gap_results = (

                    detect_research_gaps(

                        collection,

                        selected_paper

                    )

                )



            # ----------------------------------------------------

            # Display research gaps

            # ----------------------------------------------------



            st.header(

                " Research Gap Analysis"

            )



            st.markdown(

                gaps

            )



            # ----------------------------------------------------

            # Display evidence

            # ----------------------------------------------------



            gap_documents = (

                gap_results[

                    "documents"

                ][0]

            )



            if gap_documents:



                
                # Group research-gap evidence by document
                gap_metadata = gap_results.get("metadatas", [[]])[0]
                grouped_gap_sources = {}

                for metadata in gap_metadata:
                    metadata = metadata or {}
                    source = metadata.get("source", "Unknown source")
                    page = metadata.get("page_number")

                    if source not in grouped_gap_sources:
                        grouped_gap_sources[source] = set()

                    if page is not None:
                        grouped_gap_sources[source].add(page)

                if grouped_gap_sources:
                    st.header("Sources")

                    for source, pages in grouped_gap_sources.items():
                        with st.container(border=True):
                            st.markdown(f"**{source}**")

                            if pages:
                                sorted_pages = sorted(
                                    pages,
                                    key=lambda p: (
                                        not isinstance(p, (int, float)),
                                        p if isinstance(p, (int, float)) else str(p)
                                    )
                                )
                                page_text = ", ".join(
                                    str(page) for page in sorted_pages
                                )
                                st.caption(f"Relevant pages: {page_text}")
                            else:
                                st.caption("Page numbers unavailable")




            else:



                st.info(

                    "No research-gap evidence "

                    "was retrieved."

                )





    elif analysis_type == "Compare Research Papers":



        st.subheader(

            " Compare Research Papers"

        )



        st.write(

            "Select two research papers to perform "

            "a structured scientific comparison."

        )



        # --------------------------------------------------------

        # Paper selection

        # --------------------------------------------------------



        comparison_papers = st.multiselect(

            " Select exactly two papers",

            paper_names,

            max_selections=2,

            key="comparison_papers"

        )



        # --------------------------------------------------------

        # Validate selection

        # --------------------------------------------------------



        if len(comparison_papers) == 2:



            paper_a = comparison_papers[0]

            paper_b = comparison_papers[1]



            col1, col2 = st.columns(2)



            with col1:



                st.markdown(

                    "### 📄 Paper A"

                )



                st.info(

                    paper_a

                )



            with col2:



                st.markdown(

                    "### 📄 Paper B"

                )



                st.info(

                    paper_b

                )



            # ----------------------------------------------------

            # Compare button

            # ----------------------------------------------------



            if st.button(

                " Compare Papers",

                key="compare_selected_papers"

            ):



                with st.spinner(

                    "Analyzing and comparing both papers..."

                ):



                    comparison, comparison_results = (

                        compare_papers(

                            collection,

                            paper_a,

                            paper_b

                        )

                    )



                # ------------------------------------------------

                # Display comparison

                # ------------------------------------------------



                st.header(

                    " Research Paper Comparison"

                )



                st.markdown(

                    comparison

                )



                # ---------------------------------------------------------

                # Paper A evidence
                # ---------------------------------------------------------

                st.markdown("---")

                st.subheader(
                    "📄 Paper A Evidence"
                )

                paper_a_results = comparison_results[
                    "paper_a"
                ]

                paper_a_documents = paper_a_results[
                    "documents"
                ][0]

                paper_a_metadatas = paper_a_results[
                    "metadatas"
                ][0]

                if paper_a_documents:

                    
                    # Group Paper A evidence by document
                    grouped_paper_a_sources = {}

                    for metadata in paper_a_metadatas:
                        metadata = metadata or {}
                        source = metadata.get("source", paper_a)
                        page = metadata.get("page_number")

                        if source not in grouped_paper_a_sources:
                            grouped_paper_a_sources[source] = set()

                        if page is not None:
                            grouped_paper_a_sources[source].add(page)

                    for source, pages in grouped_paper_a_sources.items():
                        with st.container(border=True):
                            st.markdown(f"**{source}**")

                            if pages:
                                sorted_pages = sorted(
                                    pages,
                                    key=lambda p: (
                                        not isinstance(p, (int, float)),
                                        p if isinstance(p, (int, float)) else str(p)
                                    )
                                )
                                page_text = ", ".join(
                                    str(page) for page in sorted_pages
                                )
                                st.caption(f"Relevant pages: {page_text}")
                            else:
                                st.caption("Page numbers unavailable")


                else:

                    st.info(
                        "No evidence was retrieved "
                        "for Paper A."
                    )


                # Paper B evidence
                # ---------------------------------------------------------

                st.markdown("---")

                st.subheader(
                    "📄 Paper B Evidence"
                )

                paper_b_results = comparison_results[
                    "paper_b"
                ]

                paper_b_documents = paper_b_results[
                    "documents"
                ][0]

                paper_b_metadatas = paper_b_results[
                    "metadatas"
                ][0]

                if paper_b_documents:

                    

               
                    # Group Paper B evidence by document
                    grouped_paper_b_sources = {}

                    for metadata in paper_b_metadatas:
                        metadata = metadata or {}
                        source = metadata.get("source", paper_b)
                        page = metadata.get("page_number")

                        if source not in grouped_paper_b_sources:
                            grouped_paper_b_sources[source] = set()

                        if page is not None:
                            grouped_paper_b_sources[source].add(page)

                    for source, pages in grouped_paper_b_sources.items():
                        with st.container(border=True):
                            st.markdown(f"**{source}**")

                            if pages:
                                sorted_pages = sorted(
                                    pages,
                                    key=lambda p: (
                                        not isinstance(p, (int, float)),
                                        p if isinstance(p, (int, float)) else str(p)
                                    )
                                )
                                page_text = ", ".join(
                                    str(page) for page in sorted_pages
                                )
                                st.caption(f"Relevant pages: {page_text}")
                            else:
                                st.caption("Page numbers unavailable")


                else:

                    st.info(
                        "No evidence was retrieved "
                        "for Paper B."
                    )

        else:

            st.info(
                "Please select exactly two research papers."
            )

