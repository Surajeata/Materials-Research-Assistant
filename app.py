import streamlit as st

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





# ============================================================

# PAGE CONFIGURATION

# ============================================================



st.set_page_config(

    page_title="Materials Research Assistant",

    page_icon="",

    layout="wide"

)


# ============================================================
# CUSTOM UI STYLING — VISUAL ONLY
# ============================================================

st.markdown("""
<style>
/* Main page layout */
.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(20, 184, 166, 0.09),
            transparent 32%
        ),
        var(--background-color);
}

/* More balanced page spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* Main title */
h1 {
    font-weight: 750 !important;
    letter-spacing: -0.8px;
    padding-bottom: 0.35rem;
}

/* Section headings */
h2, h3 {
    font-weight: 650 !important;
    letter-spacing: -0.3px;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    min-height: 42px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #14b8a6;
    transform: translateY(-1px);
}

/* Text inputs */
.stTextInput input,
.stTextArea textarea {
    border-radius: 10px;
}

/* File upload areas */
[data-testid="stFileUploader"] {
    border-radius: 12px;
}

/* Expanders and tabs */
[data-testid="stExpander"] {
    border-radius: 12px;
    overflow: hidden;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 16px;
}

/* Dividers */
hr {
    opacity: 0.25;
}

/* Keep source previews readable */
blockquote {
    border-left-color: #14b8a6 !important;
    border-left-width: 3px !important;
    border-radius: 3px;
    padding-top: 4px;
    padding-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


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

                summary

            )





            # ------------------------------------------------

            # Display summary sources

            # ------------------------------------------------



            summary_documents = (

                summary_results[

                    "documents"

                ][0]

            )





            if summary_documents:



                st.header(

                    " Summary Sources"

                )





                for i in range(

                    len(

                        summary_results[

                            "ids"

                        ][0]

                    )

                ):



                    metadata = (

                        summary_results[

                            "metadatas"

                        ][0][i]

                    )





                    source = metadata[

                        "source"

                    ]



                    page = metadata[

                        "page_number"

                    ]





                    document = (

                        summary_results[

                            "documents"

                        ][0][i]

                    )





                    # ----------------------------------------

                    # Create source preview

                    # ----------------------------------------



                    preview = (

                        document

                        .replace(

                            "\n",

                            " "

                        )

                        .strip()

                    )





                    if len(preview) > 300:



                        preview = (

                            preview[:300]

                            + "..."

                        )





                    st.markdown(

                        f"""

                        **Source {i + 1}**



                        📄 **{source}**



                        📖 **Page {page}**



                       

                        """

                    )





            else:



                st.info(

                    "No summary sources were retrieved."

                )





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



                st.header(

                    " Method Evidence"

                )



                for i in range(

                    len(

                        method_results[

                            "ids"

                        ][0]

                    )

                ):



                    metadata = (

                        method_results[

                            "metadatas"

                        ][0][i]

                    )



                    source = metadata[

                        "source"

                    ]



                    page = metadata[

                        "page_number"

                    ]



                    document = (

                        method_results[

                            "documents"

                        ][0][i]

                    )



                    preview = (

                        document

                        .replace(

                            "\n",

                            " "

                        )

                        .strip()

                    )



                    if len(preview) > 300:



                        preview = (

                            preview[:300]

                            + "..."

                        )



                    st.markdown(

                        f"""

                        **Source {i + 1}**



                        📄 **{source}**



                        📖 **Page {page}**



                        

                        """

                    )



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



                st.header(

                    " Material Evidence"

                )



                for i in range(

                    len(

                        material_results[

                            "ids"

                        ][0]

                    )

                ):



                    metadata = (

                        material_results[

                            "metadatas"

                        ][0][i]

                    )



                    source = metadata[

                        "source"

                    ]



                    page = metadata[

                        "page_number"

                    ]



                    document = (

                        material_results[

                            "documents"

                        ][0][i]

                    )



                    preview = (

                        document

                        .replace(

                            "\n",

                            " "

                        )

                        .strip()

                    )



                    if len(preview) > 300:



                        preview = (

                            preview[:300]

                            + "..."

                        )



                    st.markdown(

                        f"""

                        **Source {i + 1}**



                        📄 **{source}**



                        📖 **Page {page}**



                        

                        """

                    )



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



                st.header(

                    " Finding Evidence"

                )



                for i in range(

                    len(

                        finding_results[

                            "ids"

                        ][0]

                    )

                ):



                    metadata = (

                        finding_results[

                            "metadatas"

                        ][0][i]

                    )



                    source = metadata[

                        "source"

                    ]



                    page = metadata[

                        "page_number"

                    ]



                    document = (

                        finding_results[

                            "documents"

                        ][0][i]

                    )



                    preview = (

                        document

                        .replace(

                            "\n",

                            " "

                        )

                        .strip()

                    )



                    if len(preview) > 300:



                        preview = (

                            preview[:300]

                            + "..."

                        )



                    st.markdown(

                        f"""

                        **Source {i + 1}**



                        📄 **{source}**



                        📖 **Page {page}**



                        

                        """

                    )



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



                st.header(

                    " Gap Evidence"

                )



                for i in range(

                    len(

                        gap_results[

                            "ids"

                        ][0]

                    )

                ):



                    metadata = (

                        gap_results[

                            "metadatas"

                        ][0][i]

                    )



                    source = metadata[

                        "source"

                    ]



                    page = metadata[

                        "page_number"

                    ]



                    document = (

                        gap_results[

                            "documents"

                        ][0][i]

                    )



                    preview = (

                        document

                        .replace(

                            "\n",

                            " "

                        )

                        .strip()

                    )



                    if len(preview) > 300:



                        preview = (

                            preview[:300]

                            + "..."

                        )



                    st.markdown(

                        f"""

                        **Source {i + 1}**



                        📄 **{source}**



                        📖 **Page {page}**



                        

                        """

                    )



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

                    for i in range(
                        len(paper_a_documents)
                    ):

                        metadata = paper_a_metadatas[i]

                        source = metadata.get(
                            "source",
                            paper_a
                        )

                        page = metadata.get(
                            "page_number",
                            "Unknown"
                        )

                        preview = paper_a_documents[i].strip()

                        st.markdown(
                            f"""
                **Source {i + 1}**

                📄 **{source}**

                📖 **Page {page}**

                > {preview}
                """
                        )

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

                    for i in range(
                        len(paper_b_documents)
                    ):

                        metadata = paper_b_metadatas[i]

                        source = metadata.get(
                            "source",
                            paper_b
                        )

                        page = metadata.get(
                            "page_number",
                            "Unknown"
                        )

                        preview = paper_b_documents[i].strip()

                        st.markdown(
                            f"""
                **Source {i + 1}**

                📄 **{source}**

                📖 **Page {page}**

                > {preview}
                """
                        )

                else:

                    st.info(
                        "No evidence was retrieved "
                        "for Paper B."
                    )

        else:

            st.info(
                "Please select exactly two research papers."
            )

