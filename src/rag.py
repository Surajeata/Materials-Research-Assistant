from src.retriever import search_collection
from src.llm import generate_answer
from src.retriever import (
    search_collection_for_document,
    search_hybrid_reranked
)

import re


# ============================================================
# QUESTION TYPE DETECTION
# ============================================================

def is_comparison_question(question):
    """
    Detect whether the user is asking for a comparison.
    """

    question_lower = question.lower()

    comparison_terms = [
        "compare",
        "comparison",
        "contrast",
        "difference",
        "differences",
        "differentiate",
        "versus",
        "vs",
        "which is better",
        "similarities",
        "similarity",
        "compare between"
    ]

    return any(
        term in question_lower
        for term in comparison_terms
    )


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(question):
    """
    Improve vague comparison questions before retrieval.
    """

    if not is_comparison_question(question):
        return question

    question_lower = question.lower()

    if len(question.split()) <= 12:

        return f"""
Compare the uploaded research papers in the context
of magnesium alloys and materials science.

Retrieve information relevant to comparing:

- research objectives
- material or alloy systems
- processing methods
- alloying strategies
- plastic deformability
- formability
- texture evolution
- microstructural mechanisms
- deformation mechanisms
- major findings
- conclusions

Original user question:

{question}
"""

    return question


# ============================================================
# REFERENCE CHUNK DETECTION
# ============================================================

def looks_like_reference_chunk(text):
    """
    Detect chunks that mainly contain bibliography/reference
    entries.

    This is used during paper summarization because the
    summarization process retrieves chunks directly from
    ChromaDB rather than going through normal retrieval.
    """

    if not text:
        return False


    # --------------------------------------------------------
    # Standalone numbered reference markers
    # --------------------------------------------------------

    numbered_markers = re.findall(
        r"(?m)^\s*\[\s*\d+\s*\]\s*$",
        text
    )

    if len(numbered_markers) >= 3:
        return True


    # --------------------------------------------------------
    # Reference headings
    # --------------------------------------------------------

    text_lower = text.lower()

    reference_headings = [
        "references",
        "bibliography",
        "literature cited",
        "references and notes",
        "reference list"
    ]

    beginning = text_lower[:500]

    for heading in reference_headings:

        if heading in beginning:
            return True


    # --------------------------------------------------------
    # DOI-heavy chunks
    # --------------------------------------------------------

    doi_count = len(
        re.findall(
            r"doi\s*:",
            text_lower
        )
    )

    if doi_count >= 3:
        return True


    return False


# ============================================================
# PAPER SUMMARY
# ============================================================

def summarize_paper(
    collection,
    source_name,
    max_chunks=20
):
    """
    Generate a structured summary using
    analysis-specific semantic retrieval.
    """

    # --------------------------------------------------------
    # Query specifically designed for paper summarization
    # --------------------------------------------------------

    analysis_query = """
    research paper overview objective background
    research problem materials alloys composition
    experimental approach processing methods
    microstructure texture deformation mechanisms
    results findings conclusions limitations
    future research and scientific contribution
    """

    # --------------------------------------------------------
    # Retrieve relevant evidence
    # --------------------------------------------------------

    context, results = get_analysis_context(
        collection,
        source_name,
        analysis_query,
        top_k=max_chunks
    )

    documents = results[
        "documents"
    ][0]

    # --------------------------------------------------------
    # No relevant evidence
    # --------------------------------------------------------

    if not documents:

        return (
            "No relevant content was found for this paper.",
            results
        )

    # --------------------------------------------------------
    # Prompt Gemini
    # --------------------------------------------------------

    prompt = f"""
You are an expert research assistant specializing
in materials science and engineering.

Analyze the research paper below and create a
structured technical summary.

Paper:
{source_name}

IMPORTANT RULES:

1. Use ONLY the supplied research-paper evidence.

2. Do not invent information.

3. Do not assume that information missing from
   the retrieved evidence is present in the paper.

4. Clearly distinguish reported findings from
   general scientific knowledge.

5. Mention page numbers whenever possible.

6. If a particular detail is not available,
   write:
   "Not reported in the available paper context."

Organize the answer using exactly these sections:

## 1. Research Objective

Explain:

- What problem does the paper address?
- Why is the problem important?
- What is the main research question?

## 2. Background

Explain the scientific background necessary
to understand the research.

## 3. Materials / Alloy Systems

Identify:

- Base materials
- Alloy systems
- Alloying elements
- Compositions
- Rare-earth elements
- Material conditions

## 4. Processing / Experimental Approaches

Explain:

- Processing methods
- Experimental methods
- Heat treatments
- Deformation methods
- Characterization techniques
- Important experimental conditions

## 5. Microstructure and Texture

Explain important observations involving:

- Grain structure
- Grain size
- Texture
- Recrystallization
- Precipitates
- Second phases
- Dislocations
- Twinning

## 6. Deformation Mechanisms

Explain the important mechanisms responsible
for deformation or changes in plasticity.

Only mention mechanisms supported by the evidence.

## 7. Major Findings

List the most important scientific findings.

Include numerical results when explicitly
reported.

## 8. Challenges / Limitations

Identify limitations, unresolved problems,
contradictory findings, or challenges reported
in the paper.

Do not invent limitations.

## 9. Overall Conclusion

Explain the overall scientific contribution
of the paper and what researchers can learn
from it.

## 10. Executive Summary

Finish with exactly 5 concise bullet points
containing the most important takeaways.

Use a clear technical style suitable for a
materials-science researcher.

RESEARCH PAPER EVIDENCE:

{context}
"""

    # --------------------------------------------------------
    # Generate summary
    # --------------------------------------------------------

    summary = generate_answer(
        prompt
    )

    return (
        summary,
        results
    )
    """
    Generate a structured summary of one selected research paper.

    The function:

    1. Retrieves all chunks belonging to the selected paper.
    2. Removes obvious reference/bibliography chunks.
    3. Selects chunks distributed throughout the paper.
    4. Sends the evidence to Gemini.
    5. Returns the generated summary and source information.
    """

    # --------------------------------------------------------
    # 1. Get all chunks belonging to selected paper
    # --------------------------------------------------------

    results = collection.get(
        where={
            "source": source_name
        },
        include=[
            "documents",
            "metadatas"
        ]
    )


    documents = results.get(
        "documents",
        []
    )

    metadatas = results.get(
        "metadatas",
        []
    )

    ids = results.get(
        "ids",
        []
    )


    # --------------------------------------------------------
    # 2. Check whether paper exists
    # --------------------------------------------------------

    if not documents:

        return (
            "I could not find the selected paper "
            "in the research database.",
            {
                "documents": [[]],
                "metadatas": [[]],
                "ids": [[]]
            }
        )


    # --------------------------------------------------------
    # 3. Build clean chunk list
    # --------------------------------------------------------

    chunks = []


    for i in range(
        len(documents)
    ):

        text = documents[i]

        metadata = metadatas[i]

        # Skip obvious bibliography chunks
        if looks_like_reference_chunk(text):
            continue


        chunks.append({
            "id": ids[i],
            "text": text,
            "metadata": metadata
        })


    # --------------------------------------------------------
    # 4. Sort chunks by page and chunk ID
    #
    # This gives us approximately the natural order
    # of the research paper.
    # --------------------------------------------------------

    chunks.sort(
        key=lambda chunk: (
            chunk["metadata"].get(
                "page_number",
                0
            ),
            chunk["id"]
        )
    )


    # --------------------------------------------------------
    # 5. Select representative chunks
    #
    # We don't send every chunk from a 29-page paper
    # because that would create unnecessarily large prompts.
    #
    # Instead we distribute the selected chunks throughout
    # the paper so the summary gets broad coverage.
    # --------------------------------------------------------

    if len(chunks) > max_chunks:

        step = (
            len(chunks) / max_chunks
        )

        selected_chunks = []

        for i in range(max_chunks):

            index = int(
                i * step
            )

            selected_chunks.append(
                chunks[index]
            )

    else:

        selected_chunks = chunks


    # --------------------------------------------------------
    # 6. Build summary context
    # --------------------------------------------------------

    context_parts = []


    for i, chunk in enumerate(
        selected_chunks
    ):

        metadata = chunk["metadata"]

        context_parts.append(
            f"""
EVIDENCE {i + 1}

Paper:
{metadata.get("source", source_name)}

Page:
{metadata.get("page_number", "Unknown")}

Content:
{chunk["text"]}
"""
        )


    context = "\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # 7. Create summarization prompt
    # --------------------------------------------------------

    prompt = f"""
You are an expert research assistant specializing
in materials science, metallurgy, and magnesium alloys.

Create a structured summary of the research paper
using ONLY the evidence provided below.

Do NOT use outside knowledge.

Do NOT invent information.

If a particular section is not sufficiently supported
by the provided evidence, say that the information
was not available in the retrieved excerpts.

The summary should contain the following sections:

## 1. Research Objective

Explain what problem or research topic the paper addresses.

## 2. Background

Briefly explain the scientific context discussed
in the paper.

## 3. Materials / Alloy Systems

Identify the materials, magnesium alloys,
elements, or alloy systems explicitly mentioned
in the evidence.

## 4. Processing / Experimental Approaches

Describe the important processing, deformation,
heat-treatment, or experimental approaches
discussed in the evidence.

## 5. Microstructure and Texture

Explain important microstructural or texture-related
mechanisms discussed by the authors.

## 6. Deformation Mechanisms

Explain the important deformation mechanisms,
slip systems, twinning, CRSS, or related mechanisms
when supported by the evidence.

## 7. Major Findings

Summarize the most important findings or conclusions.

## 8. Challenges / Limitations

Mention limitations, difficulties, or unresolved
problems explicitly discussed in the evidence.

## 9. Overall Conclusion

Give a concise overall conclusion based only
on the provided evidence.

IMPORTANT CITATION RULE:

Whenever possible, mention the page number supporting
a statement.

Example:

"Texture weakening can improve formability (Page 5)."

Do not invent page numbers.

Research paper:

{source_name}

Research-paper evidence:
--------------------------------

{context}

--------------------------------

Produce a technically accurate and well-structured
research summary.
"""


    # --------------------------------------------------------
    # 8. Generate summary
    # --------------------------------------------------------

    answer = generate_answer(
        prompt
    )


    # --------------------------------------------------------
    # 9. Prepare sources for UI
    # --------------------------------------------------------

    source_results = {

        "documents": [[
            chunk["text"]
            for chunk in selected_chunks
        ]],

        "metadatas": [[
            chunk["metadata"]
            for chunk in selected_chunks
        ]],

        "ids": [[
            chunk["id"]
            for chunk in selected_chunks
        ]]
    }


    return answer, source_results


# ============================================================
# NORMAL RAG QUESTION
# ============================================================

def answer_question(
    collection,
    question,
    top_k=5
):
    """
    Retrieve relevant research-paper chunks
    and use Gemini to generate a grounded answer.
    """

    # --------------------------------------------------------
    # 1. Determine question type
    # --------------------------------------------------------

    comparison_mode = is_comparison_question(
        question
    )


    # --------------------------------------------------------
    # 2. Improve query before retrieval
    # --------------------------------------------------------

    retrieval_query = expand_query(
        question
    )


    # --------------------------------------------------------
    # 3. Retrieve relevant chunks
    # --------------------------------------------------------

    # 3. Retrieve relevant chunks using hybrid retrieval
# and CrossEncoder reranking.

    results = search_hybrid_reranked(
        collection=collection,
        query=retrieval_query,
        top_k=top_k,
        candidate_k=max(top_k * 8, 40),
        semantic_weight=0.6,
        bm25_weight=0.4,
        rerank_query=question
)


    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    ids = results["ids"][0]


    # --------------------------------------------------------
    # 4. Handle no relevant information
    # --------------------------------------------------------

    if not documents:

        answer = (
            "I could not find sufficiently relevant "
            "information in the uploaded research papers "
            "to answer this question."
        )

        return answer, results


    # --------------------------------------------------------
    # 5. Build research context
    # --------------------------------------------------------

    context_parts = []


    for i in range(
        len(documents)
    ):

        context_parts.append(
            f"""
SOURCE {i + 1}

Paper:
{metadatas[i]["source"]}

Page:
{metadatas[i]["page_number"]}

Chunk ID:
{ids[i]}

Content:
{documents[i]}
"""
        )


    context = "\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # 6. Comparison prompt
    # --------------------------------------------------------

    if comparison_mode:

        prompt = f"""
You are a research assistant specializing in
materials science and magnesium alloys.

The user is asking for a COMPARISON of research papers.

Answer the question using ONLY the research-paper
excerpts provided below.

Do not use outside knowledge.

Do not invent facts.

If information about a particular comparison aspect
is not available in the provided excerpts, explicitly
say that it was not available in the retrieved sources.

IMPORTANT:

Identify which paper supports each claim.

Mention the relevant page number whenever possible.

When comparing papers, distinguish clearly between
information from different papers.

Organize the answer using a comparison table whenever
appropriate.

Useful comparison aspects may include:

1. Research objective
2. Materials or alloy systems
3. Alloying strategies
4. Processing methods
5. Deformation mechanisms
6. Texture evolution
7. Microstructural mechanisms
8. Strategies for improving plastic deformability
9. Major findings
10. Overall conclusions

Do not force every category into the answer if the
retrieved sources do not contain enough information.

Research-paper excerpts:
--------------------------------

{context}

--------------------------------

User question:

{question}

--------------------------------

Provide a technically accurate and concise
research-paper comparison.
"""


    # --------------------------------------------------------
    # 7. Normal question prompt
    # --------------------------------------------------------

    else:

        prompt = f"""
You are a research assistant specializing in materials science.

Answer the user's question using ONLY the information
provided in the research-paper excerpts below.

If the excerpts do not contain enough information to answer
the question, clearly say that the information is not
available in the provided sources.

Do not invent facts.

When making a claim, mention the relevant page number.

Research-paper excerpts:
--------------------------------

{context}

--------------------------------

User question:

{question}

Provide a clear and technically accurate answer.
"""


    # --------------------------------------------------------
    # 8. Generate answer
    # --------------------------------------------------------

    answer = generate_answer(
        prompt
    )


    return answer, results

def summarize_paper(
    collection,
    source_name,
    max_chunks=24
):
    """
    Generate a structured summary of a selected research paper.
    """

    # --------------------------------------------------------
    # Get all chunks belonging to the selected paper
    # --------------------------------------------------------

    results = collection.get(
        where={
            "source": source_name
        },
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = results.get(
        "documents",
        []
    )

    metadatas = results.get(
        "metadatas",
        []
    )

    ids = results.get(
        "ids",
        []
    )

    # --------------------------------------------------------
    # No document found
    # --------------------------------------------------------

    if not documents:

        return (
            "No content was found for this paper.",
            {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]]
            }
        )

    # --------------------------------------------------------
    # Remove reference/bibliography chunks
    # --------------------------------------------------------

    filtered_chunks = []

    for i in range(len(documents)):

        metadata = metadatas[i]

        # Metadata-based reference filtering
        if metadata.get(
            "is_reference",
            False
        ):
            continue

        text = documents[i]

        # Additional safety check
        if looks_like_reference_chunk(text):
            continue

        filtered_chunks.append({
            "id": ids[i],
            "document": text,
            "metadata": metadata
        })

    # --------------------------------------------------------
    # Sort chunks by page number
    # --------------------------------------------------------

    filtered_chunks.sort(
        key=lambda x: (
            x["metadata"].get(
                "page_number",
                0
            ),
            x["id"]
        )
    )

    # --------------------------------------------------------
    # Select representative chunks
    # --------------------------------------------------------

    if len(filtered_chunks) > max_chunks:

        step = (
            len(filtered_chunks)
            / max_chunks
        )

        selected_chunks = []

        for i in range(max_chunks):

            index = int(
                i * step
            )

            selected_chunks.append(
                filtered_chunks[index]
            )

    else:

        selected_chunks = filtered_chunks

    # --------------------------------------------------------
    # Build research context
    # --------------------------------------------------------

    context_parts = []

    for chunk in selected_chunks:

        page = chunk[
            "metadata"
        ].get(
            "page_number",
            "Unknown"
        )

        text = chunk[
            "document"
        ]

        context_parts.append(
            f"""
--- Page {page} ---

{text}
"""
        )

    context = "\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Prompt Gemini
    # --------------------------------------------------------

    prompt = f"""
You are a research assistant specializing in
materials science and engineering.

Analyze the following research paper and create
a structured technical summary.

Paper:
{source_name}

Use ONLY the information provided in the context.

Do not invent information.

Where possible, mention the relevant page number
for important claims.

Structure the answer using the following sections:

## 1. Research Objective

Explain what problem the paper investigates
and why the research is important.

## 2. Background

Explain the scientific background needed
to understand the paper.

## 3. Materials / Alloy Systems

Identify the materials, alloys, compositions,
or material systems discussed.

## 4. Processing / Experimental Approaches

Explain the experimental methods, processing
routes, characterization techniques, simulations,
or other approaches used.

## 5. Microstructure and Texture

Explain important microstructural features,
texture evolution, grain structure, phases,
or related observations.

## 6. Deformation Mechanisms

Explain the mechanisms responsible for deformation,
plasticity, strengthening, softening, or
improved formability.

## 7. Major Findings

List the most important findings and conclusions
of the research.

## 8. Challenges / Limitations

Identify limitations, unresolved problems,
or challenges discussed or apparent from
the paper.

## 9. Overall Conclusion

Give a concise overall conclusion explaining
the main contribution of the paper.

Keep the answer technical but easy to understand.

Research paper context:

{context}
"""

    # --------------------------------------------------------
    # Generate summary
    # --------------------------------------------------------

    summary = generate_answer(
        prompt
    )

    # --------------------------------------------------------
    # Prepare source information
    # --------------------------------------------------------

    summary_results = {
        "ids": [[
            chunk["id"]
            for chunk in selected_chunks
        ]],

        "documents": [[
            chunk["document"]
            for chunk in selected_chunks
        ]],

        "metadatas": [[
            chunk["metadata"]
            for chunk in selected_chunks
        ]]
    }

    return (
        summary,
        summary_results
    )


def extract_experimental_methods(
    collection,
    source_name,
    max_chunks=20
):
    """
    Extract experimental and processing methods
    using analysis-specific semantic retrieval.
    """

    # --------------------------------------------------------
    # Query specifically designed for experimental methods
    # --------------------------------------------------------

    analysis_query = """
    experimental methodology sample preparation
    material preparation alloy preparation
    casting homogenization heat treatment
    extrusion rolling forging processing
    severe plastic deformation
    deformation temperature strain rate
    strain compression tensile testing
    mechanical testing characterization
    microscopy SEM TEM EBSD XRD EDS
    experimental procedure processing parameters
    """

    # --------------------------------------------------------
    # Retrieve relevant evidence
    # --------------------------------------------------------

    context, results = get_analysis_context(
        collection,
        source_name,
        analysis_query,
        top_k=max_chunks
    )

    documents = results[
        "documents"
    ][0]

    # --------------------------------------------------------
    # No relevant evidence
    # --------------------------------------------------------

    if not documents:

        return (
            "No experimental-method information "
            "was found for this paper.",
            results
        )

    # --------------------------------------------------------
    # Prompt Gemini
    # --------------------------------------------------------

    prompt = f"""
You are an expert research assistant specializing
in materials science and engineering.

Analyze the following research paper and extract
its experimental and processing methodology.

Paper:
{source_name}

IMPORTANT RULES:

1. Use ONLY the supplied research-paper evidence.

2. Do not invent experimental parameters.

3. Clearly distinguish between:
   - methods performed by the paper's authors
   - methods reported from previous studies
   - methods discussed generally

4. If a parameter is not available, write:
   "Not reported in the available paper context."

5. Mention page numbers whenever possible.

6. Do not assume that a processing condition applies
   to every alloy or experiment discussed.

Organize the answer using exactly these sections:

## 1. Materials / Alloy Used

Identify:

- Base material
- Alloy designation
- Alloy composition
- Alloying elements
- Material condition

Clearly indicate whether the material was
experimentally investigated or discussed from
previous literature.

## 2. Sample Preparation

Extract reported information about:

- Sample dimensions
- Cutting
- Machining
- Surface preparation
- Homogenization
- Heat treatment
- Pre-processing

Include temperatures and times when reported.

## 3. Processing Methods

Identify the major processing techniques.

Consider:

- Casting
- Extrusion
- Rolling
- Forging
- Annealing
- Heat treatment
- Severe plastic deformation
- Other processing methods

For each method provide reported parameters such as:

- Temperature
- Time
- Reduction
- Speed
- Strain
- Number of passes
- Strain rate

Do not invent missing values.

## 4. Deformation / Mechanical Testing

Identify:

- Tensile testing
- Compression testing
- Formability testing
- Other mechanical tests
- Deformation temperature
- Strain rate
- Applied strain
- Loading direction
- Testing conditions

Include numerical values only when explicitly
supported by the evidence.

## 5. Characterization Techniques

Identify techniques such as:

- Optical microscopy
- SEM
- TEM
- EBSD
- XRD
- EDS
- DSC
- Hardness testing
- Mechanical testing
- Other characterization methods

For each technique explain what it was used
to investigate.

## 6. Experimental Workflow

Provide a step-by-step description of the
experimental procedure when the evidence
supports one.

For example:

Material preparation
→ Processing
→ Heat treatment
→ Deformation/testing
→ Characterization
→ Analysis

Do not construct steps that are not supported
by the paper.

## 7. Important Experimental Parameters

Create a table:

| Parameter | Value | Material / Condition | Page |
|---|---|---|---|

Include only explicitly reported parameters.

## 8. Methods Most Important to the Results

Explain which experimental or processing
methods were most important for obtaining
the reported findings.

Support the explanation with evidence.

## 9. Methodological Limitations

Identify any limitations or missing
experimental information explicitly discussed
in the paper.

Do not invent limitations.

## 10. Experimental Method Summary

Finish with exactly 5 concise bullet points
summarizing the most important methodological
details.

RESEARCH PAPER EVIDENCE:

{context}
"""

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    answer = generate_answer(
        prompt
    )

    return (
        answer,
        results
    )
    """
    Extract experimental and processing methods
    from a selected research paper.
    """

    # --------------------------------------------------------
    # Get chunks belonging to selected paper
    # --------------------------------------------------------

    results = collection.get(
        where={
            "source": source_name
        },
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = results.get(
        "documents",
        []
    )

    metadatas = results.get(
        "metadatas",
        []
    )

    ids = results.get(
        "ids",
        []
    )

    # --------------------------------------------------------
    # No document found
    # --------------------------------------------------------

    if not documents:

        return (
            "No content was found for this paper.",
            {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]]
            }
        )

    # --------------------------------------------------------
    # Remove reference chunks
    # --------------------------------------------------------

    filtered_chunks = []

    for i in range(len(documents)):

        metadata = metadatas[i]

        if metadata.get(
            "is_reference",
            False
        ):
            continue

        text = documents[i]

        if looks_like_reference_chunk(text):
            continue

        filtered_chunks.append({
            "id": ids[i],
            "document": text,
            "metadata": metadata
        })

    # --------------------------------------------------------
    # Sort by page
    # --------------------------------------------------------

    filtered_chunks.sort(
        key=lambda x: (
            x["metadata"].get(
                "page_number",
                0
            ),
            x["id"]
        )
    )

    # --------------------------------------------------------
    # Select representative chunks
    # --------------------------------------------------------

    if len(filtered_chunks) > max_chunks:

        step = (
            len(filtered_chunks)
            / max_chunks
        )

        selected_chunks = []

        for i in range(max_chunks):

            index = int(
                i * step
            )

            selected_chunks.append(
                filtered_chunks[index]
            )

    else:

        selected_chunks = filtered_chunks

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for chunk in selected_chunks:

        page = chunk[
            "metadata"
        ].get(
            "page_number",
            "Unknown"
        )

        text = chunk[
            "document"
        ]

        context_parts.append(
            f"""
--- Page {page} ---

{text}
"""
        )

    context = "\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Prompt Gemini
    # --------------------------------------------------------

    prompt = f"""
You are an expert research assistant specializing
in materials science and engineering.

Analyze the following research paper and extract
the experimental and processing methodology.

Paper:
{source_name}

Use ONLY the information present in the supplied
research-paper context.

Do not invent missing experimental parameters.

If a parameter is not reported in the available
context, explicitly write:

"Not reported in the available paper context."

Organize the answer using the following structure:

## 1. Materials / Alloy Used

Identify:

- Base material
- Alloy composition
- Alloying elements
- Material condition
- Initial microstructure, if reported

## 2. Sample Preparation

Explain:

- Sample dimensions
- Cutting/preparation
- Homogenization
- Heat treatment
- Pre-processing
- Other preparation steps

## 3. Processing Methods

Identify all major processing methods, such as:

- Casting
- Rolling
- Extrusion
- Forging
- Heat treatment
- Severe plastic deformation
- Other processing techniques

For each method, provide important parameters
such as temperature, reduction, speed, strain,
strain rate, passes, or other reported conditions.

## 4. Deformation / Mechanical Testing

Identify:

- Type of mechanical test
- Tensile/compression testing
- Formability testing
- Deformation temperature
- Strain rate
- Applied strain
- Testing direction
- Other relevant parameters

## 5. Characterization Techniques

Identify techniques such as:

- Optical microscopy
- SEM
- TEM
- EBSD
- XRD
- EDS
- DSC
- Hardness testing
- Mechanical testing
- Other techniques

Explain what each technique was used to investigate.

## 6. Experimental Workflow

Give a step-by-step description of the overall
experimental procedure from material preparation
to characterization and analysis.

## 7. Important Experimental Parameters

Create a table with:

| Parameter | Value | Page |
|---|---|---|

Include only parameters explicitly reported
in the paper.

## 8. Key Methodological Takeaways

Briefly explain which experimental methods were
most important for obtaining the paper's results.

Mention page numbers wherever possible.

Research paper context:

{context}
"""

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    answer = generate_answer(
        prompt
    )

    # --------------------------------------------------------
    # Prepare source information
    # --------------------------------------------------------

    method_results = {
        "ids": [[
            chunk["id"]
            for chunk in selected_chunks
        ]],

        "documents": [[
            chunk["document"]
            for chunk in selected_chunks
        ]],

        "metadatas": [[
            chunk["metadata"]
            for chunk in selected_chunks
        ]]
    }

    return (
        answer,
        method_results
    )

def extract_materials_and_alloys(
    collection,
    source_name,
    max_chunks=20
):
    """
    Extract materials and alloy-related information
    using analysis-specific semantic retrieval.
    """

    analysis_query = """
    materials and alloy systems studied in the research paper
    magnesium alloys Mg alloy composition
    alloying elements concentration composition
    rare earth elements RE additions
    Zn Al Mn Ca Sn Zr Y Gd Nd Dy Ce La
    base material alloy designation
    chemical composition
    processing conditions
    casting homogenization heat treatment
    extrusion rolling forging deformation processing
    microstructure grain size texture
    precipitates phases intermetallic compounds
    mechanical properties strength ductility
    plastic deformability formability
    effect of alloying elements on microstructure
    effect of composition on deformation behavior
    material processing structure property relationship
    """

    context, results = get_analysis_context(
        collection,
        source_name,
        analysis_query,
        top_k=max_chunks
    )

    documents = results[
        "documents"
    ][0]

    if not documents:

        return (
            "No materials or alloy information "
            "was found for this paper.",
            results
        )

    prompt = f"""
You are an expert research assistant specializing
in materials science, metallurgy, and magnesium alloys.

Analyze the following research paper and extract
all important information related to materials,
alloy systems, compositions, alloying elements,
processing conditions, microstructure, and properties.

Paper:
{source_name}

IMPORTANT RULES:

1. Use ONLY the supplied research-paper evidence.

2. Do not invent alloy compositions, processing
   parameters, phases, properties, or experimental
   results.

3. Clearly distinguish between:
   - materials actually studied by the paper
   - materials discussed from previous research
   - general materials-science background

4. If an exact composition or parameter is not
   available, write:
   "Not reported in the available paper context."

5. Mention page numbers whenever possible.

6. Pay particular attention to magnesium alloys,
   alloying elements, rare-earth additions,
   processing routes, texture, microstructure,
   deformation behavior, and mechanical properties.

7. Do not assume that a processing condition or
   composition applies to every alloy discussed.

8. If the paper is a review article, clearly
   distinguish between individual alloy systems
   reported across different studies.

Organize the answer using exactly these sections:

## 1. Base Materials

Identify the main base materials discussed.

For magnesium research, identify the magnesium
material/system and relevant alloy families.

---

## 2. Alloy Systems

Create a structured table containing:

| Alloy / Material | Composition | Alloying Elements | Purpose / Role | Source Context |

If the exact composition is unavailable, state:

"Not reported in the available paper context."

---

## 3. Alloying Elements

Create a table:

| Element | Role in Alloy | Reported Effect | Relevant Microstructural / Deformation Effect |

Focus particularly on:

- Al
- Zn
- Mn
- Ca
- Sn
- Zr
- Y
- Gd
- Nd
- Dy
- Ce
- La
- other rare-earth elements

Only include elements supported by the paper evidence.

---

## 4. Rare-Earth Elements

Identify the rare-earth elements discussed.

For each one, explain:

- why it is added
- effect on texture
- effect on grain structure
- effect on recrystallization
- effect on deformation mechanisms
- effect on plastic deformability
- effect on mechanical properties

Do not add information that is not supported
by the supplied evidence.

---

## 5. Processing Conditions

Extract important processing routes and conditions.

Include, where available:

- casting
- homogenization
- heat treatment
- extrusion
- rolling
- forging
- annealing
- severe plastic deformation
- deformation temperature
- strain
- strain rate
- processing sequence

Mention exact values only when reported.

---

## 6. Microstructure

Extract information about:

- grain size
- grain morphology
- recrystallization
- precipitates
- second phases
- intermetallic phases
- twins
- dislocations
- texture
- grain refinement

Explain how processing or alloying changes the
microstructure.

---

## 7. Mechanical / Deformation Properties

Extract reported effects on:

- yield strength
- tensile strength
- ductility
- elongation
- plastic deformability
- formability
- fracture behavior
- deformation mechanisms

Include numerical values only when supported
by the evidence.

---

## 8. Material → Processing → Structure → Property Relationship

Explain the chain:

Material / Composition
        ↓
Processing
        ↓
Microstructure / Texture
        ↓
Deformation Mechanism
        ↓
Mechanical Properties / Plastic Deformability

Use evidence from the paper to explain the
relationships.

---

## 9. Consolidated Materials Table

Create a final concise table:

| Material / Alloy | Composition | Processing | Microstructure / Texture | Main Property / Effect |

Only include information supported by the paper.

---

## 10. Key Materials-Science Takeaways

Give the most important materials-related
scientific conclusions from the paper.

Focus on:

- alloy design
- rare-earth additions
- processing
- texture modification
- microstructure
- deformation mechanisms
- plastic deformability

Finish with exactly 5 concise bullet points.

RESEARCH PAPER EVIDENCE:

{context}
"""

    answer = generate_answer(
        prompt
    )

    return (
        answer,
        results
    )
    """
    Extract materials, alloy compositions,
    processing conditions, microstructure,
    and related material information from
    a selected research paper.
    """

    # --------------------------------------------------------
    # Get chunks belonging to selected paper
    # --------------------------------------------------------

    results = collection.get(
        where={
            "source": source_name
        },
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = results.get(
        "documents",
        []
    )

    metadatas = results.get(
        "metadatas",
        []
    )

    ids = results.get(
        "ids",
        []
    )

    # --------------------------------------------------------
    # No document found
    # --------------------------------------------------------

    if not documents:

        return (
            "No content was found for this paper.",
            {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]]
            }
        )

    # --------------------------------------------------------
    # Remove reference chunks
    # --------------------------------------------------------

    filtered_chunks = []

    for i in range(len(documents)):

        metadata = metadatas[i]

        if metadata.get(
            "is_reference",
            False
        ):
            continue

        text = documents[i]

        if looks_like_reference_chunk(text):
            continue

        filtered_chunks.append({
            "id": ids[i],
            "document": text,
            "metadata": metadata
        })

    # --------------------------------------------------------
    # Sort by page
    # --------------------------------------------------------

    filtered_chunks.sort(
        key=lambda x: (
            x["metadata"].get(
                "page_number",
                0
            ),
            x["id"]
        )
    )

    # --------------------------------------------------------
    # Select representative chunks
    # --------------------------------------------------------

    if len(filtered_chunks) > max_chunks:

        step = (
            len(filtered_chunks)
            / max_chunks
        )

        selected_chunks = []

        for i in range(max_chunks):

            index = int(
                i * step
            )

            selected_chunks.append(
                filtered_chunks[index]
            )

    else:

        selected_chunks = filtered_chunks

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for chunk in selected_chunks:

        page = chunk[
            "metadata"
        ].get(
            "page_number",
            "Unknown"
        )

        text = chunk[
            "document"
        ]

        context_parts.append(
            f"""
--- Page {page} ---

{text}
"""
        )

    context = "\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Prompt Gemini
    # --------------------------------------------------------

    prompt = f"""
You are an expert materials-science research assistant.

Analyze the following research paper and extract
structured information about the materials and
alloy systems discussed.

Paper:
{source_name}

IMPORTANT RULES:

1. Use ONLY information contained in the supplied
   research-paper context.

2. Do not invent alloy compositions, percentages,
   processing conditions, phases, or properties.

3. Clearly distinguish between:
   - materials actually used in experiments
   - materials mentioned from previous studies
   - materials discussed theoretically

4. If information is unavailable, write:
   "Not reported in the available paper context."

5. Include page numbers whenever possible.

Organize the answer as follows:

## 1. Base Materials

Identify the main base materials discussed.

For each material provide:

- Material
- Type
- Application/research purpose
- Page

## 2. Alloy Systems

Identify all important alloy systems.

For each alloy provide:

- Alloy designation
- Base element
- Alloying elements
- Composition
- Page

Pay special attention to magnesium alloys
and rare-earth-containing alloys.

## 3. Alloying Elements

Create a table:

| Element | Role / Purpose | Reported Concentration | Page |
|---|---|---|---|

Explain why each important alloying element
was added or discussed, if the paper states it.

## 4. Rare-Earth Elements

If rare-earth elements are present, identify:

- Element
- Symbol
- Concentration
- Purpose/effect
- Relevant phase or microstructural effect
- Page

If no rare-earth elements are discussed,
state that clearly.

## 5. Processing Conditions

For each important material or alloy,
identify reported processing conditions such as:

- Casting
- Homogenization
- Extrusion
- Rolling
- Forging
- Heat treatment
- Annealing
- Severe plastic deformation

Include temperatures, times, reductions,
speeds, passes, or other parameters when reported.

## 6. Microstructure

Extract reported information about:

- Grain size
- Grain morphology
- Recrystallization
- Texture
- Precipitates
- Second phases
- Intermetallic phases
- Dislocations
- Twins
- Other important microstructural features

Include the associated alloy/material and page.

## 7. Mechanical / Deformation Properties

Extract reported properties such as:

- Yield strength
- Ultimate tensile strength
- Elongation
- Hardness
- Ductility
- Formability
- Plastic strain
- Deformation behavior

Include numerical values only when explicitly
reported in the context.

## 8. Material → Processing → Structure → Property Relationship

Explain the major relationships identified
in the paper.

For example:

Alloying
    ↓
Microstructural change
    ↓
Texture / deformation mechanism
    ↓
Mechanical property or formability change

Use specific examples from the paper.

## 9. Consolidated Materials Table

Create a final table:

| Material / Alloy | Composition | Processing | Microstructure | Key Property / Behavior | Page |
|---|---|---|---|---|---|

Only include information explicitly supported
by the paper.

## 10. Key Materials-Science Takeaways

Give the 5–10 most important materials-science
insights from the paper.

Research paper context:

{context}
"""

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    answer = generate_answer(
        prompt
    )

    # --------------------------------------------------------
    # Prepare source information
    # --------------------------------------------------------

    material_results = {
        "ids": [[
            chunk["id"]
            for chunk in selected_chunks
        ]],

        "documents": [[
            chunk["document"]
            for chunk in selected_chunks
        ]],

        "metadatas": [[
            chunk["metadata"]
            for chunk in selected_chunks
        ]]
    }

    return (
        answer,
        material_results
    )

def extract_key_findings(
    collection,
    source_name,
    max_chunks=20
):
    """
    Extract the most important scientific findings
    using analysis-specific semantic retrieval.
    """

    analysis_query = """
    major scientific findings results conclusions
    important experimental results
    quantitative results numerical findings
    magnesium alloy plastic deformability
    ductility strength elongation mechanical properties
    deformation behavior deformation mechanisms
    processing effects on microstructure
    processing effects on texture
    microstructure-property relationship
    alloy composition effects
    alloying element effects
    rare earth element effects
    grain refinement recrystallization
    texture modification basal texture
    twinning slip mechanisms
    improvements in deformability
    limitations contradictory findings
    scientific contribution
    """

    context, results = get_analysis_context(
        collection,
        source_name,
        analysis_query,
        top_k=max_chunks
    )

    documents = results[
        "documents"
    ][0]

    if not documents:

        return (
            "No relevant scientific findings "
            "were found for this paper.",
            results
        )

    prompt = f"""
You are an expert research assistant specializing
in materials science, metallurgy, and magnesium alloys.

Analyze the following research paper and extract
its most important scientific findings.

Paper:
{source_name}

IMPORTANT RULES:

1. Use ONLY the supplied research-paper evidence.

2. Do not invent results, numerical values,
   mechanisms, or conclusions.

3. Clearly distinguish between:
   - findings directly reported by the authors
   - findings reported from previous studies
   - general scientific background

4. If a numerical result or specific finding is
   not available in the supplied evidence, write:
   "Not reported in the available paper context."

5. Mention page numbers whenever possible.

6. Give priority to scientifically meaningful
   results rather than background information.

7. Pay particular attention to relationships between:

   Alloy Composition
        ↓
   Processing
        ↓
   Microstructure / Texture
        ↓
   Deformation Mechanism
        ↓
   Mechanical Properties / Plastic Deformability

8. If the paper is a review article, do not present
   findings from different studies as if they were
   results from one experiment.

9. Do not assume that a reported improvement applies
   to all magnesium alloys.

Organize the answer using exactly these sections:

## 1. Main Scientific Findings

List the most important scientific findings.

For each finding explain:

- What was observed?
- In which material/alloy system?
- Under what processing or experimental condition?
- What was the scientific significance?

Mention page numbers whenever possible.

---

## 2. Important Quantitative Results

Create a table:

| Material / Condition | Property / Parameter | Reported Value | Effect / Interpretation | Page |

Only include numerical values explicitly supported
by the supplied evidence.

---

## 3. Processing → Microstructure Findings

Explain how processing conditions affected:

- grain size
- recrystallization
- precipitates
- second phases
- twins
- dislocations
- texture
- grain morphology

Explain the scientific significance of these changes.

---

## 4. Microstructure → Property Findings

Explain how changes in microstructure and texture
affected:

- strength
- ductility
- elongation
- plastic deformability
- formability
- fracture behavior

Focus on cause-and-effect relationships supported
by the paper.

---

## 5. Deformation Mechanisms

Identify the important deformation mechanisms
reported in the paper.

Consider:

- basal slip
- non-basal slip
- prismatic slip
- pyramidal slip
- twinning
- grain boundary processes
- dynamic recrystallization
- dislocation activity

Only include mechanisms supported by the evidence.

---

## 6. Alloying / Composition Effects

Explain how alloying elements or changes in
composition affected the reported behavior.

Pay particular attention to:

- rare-earth elements
- Zn
- Al
- Mn
- Ca
- Sn
- Zr
- Y
- Gd
- Nd
- Dy
- Ce
- La

Only discuss elements actually supported by
the supplied evidence.

---

## 7. Improvements Compared With Reference Conditions

Identify cases where the paper reports improvements
in:

- ductility
- strength
- plastic deformability
- formability
- texture
- microstructure
- processing performance

For every improvement, identify the relevant
material and comparison condition when available.

Do not invent a baseline.

---

## 8. Limitations / Contradictory Findings

Identify:

- limitations
- inconsistent results
- contradictory findings
- conditions where an improvement was not observed
- limitations of existing processing methods
- limitations of the available evidence

Clearly distinguish author-reported limitations
from observations inferred from the supplied evidence.

---

## 9. Most Important Scientific Contribution

Explain the single most important scientific
contribution or insight of the paper.

Then explain why it matters for:

- magnesium alloy design
- processing
- texture control
- deformation behavior
- plastic deformability

---

## 10. Executive Summary

Finish with exactly 5 concise bullet points.

Each bullet should contain one important
scientific finding.

RESEARCH PAPER EVIDENCE:

{context}
"""

    answer = generate_answer(
        prompt
    )

    return (
        answer,
        results
    )
    """
    Extract the major scientific findings,
    quantitative results, mechanisms, and
    conclusions from a selected research paper.
    """

    # --------------------------------------------------------
    # Get chunks belonging to selected paper
    # --------------------------------------------------------

    results = collection.get(
        where={
            "source": source_name
        },
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = results.get(
        "documents",
        []
    )

    metadatas = results.get(
        "metadatas",
        []
    )

    ids = results.get(
        "ids",
        []
    )

    # --------------------------------------------------------
    # No document found
    # --------------------------------------------------------

    if not documents:

        return (
            "No content was found for this paper.",
            {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]]
            }
        )

    # --------------------------------------------------------
    # Remove reference chunks
    # --------------------------------------------------------

    filtered_chunks = []

    for i in range(len(documents)):

        metadata = metadatas[i]

        if metadata.get(
            "is_reference",
            False
        ):
            continue

        text = documents[i]

        if looks_like_reference_chunk(text):
            continue

        filtered_chunks.append({
            "id": ids[i],
            "document": text,
            "metadata": metadata
        })

    # --------------------------------------------------------
    # Sort chunks by page
    # --------------------------------------------------------

    filtered_chunks.sort(
        key=lambda x: (
            x["metadata"].get(
                "page_number",
                0
            ),
            x["id"]
        )
    )

    # --------------------------------------------------------
    # Select representative chunks
    # --------------------------------------------------------

    if len(filtered_chunks) > max_chunks:

        step = (
            len(filtered_chunks)
            / max_chunks
        )

        selected_chunks = []

        for i in range(max_chunks):

            index = int(
                i * step
            )

            selected_chunks.append(
                filtered_chunks[index]
            )

    else:

        selected_chunks = filtered_chunks

    # --------------------------------------------------------
    # Build research context
    # --------------------------------------------------------

    context_parts = []

    for chunk in selected_chunks:

        page = chunk[
            "metadata"
        ].get(
            "page_number",
            "Unknown"
        )

        text = chunk[
            "document"
        ]

        context_parts.append(
            f"""
--- Page {page} ---

{text}
"""
        )

    context = "\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Prompt Gemini
    # --------------------------------------------------------

    prompt = f"""
You are an expert materials-science research
analyst.

Analyze the following research paper and extract
its most important scientific findings.

Paper:
{source_name}

Use ONLY the information provided in the
research-paper context.

Do not invent results, numerical values,
mechanisms, or conclusions.

If a result is not explicitly available,
write:

"Not reported in the available paper context."

Always distinguish between:

- Findings directly demonstrated by the paper
- Findings reported from previous studies
- Interpretations or proposed mechanisms

Mention page numbers whenever possible.

Organize the answer using the following structure:

## 1. Main Scientific Findings

List the 5–10 most important findings of the paper.

For each finding explain:

- What was observed
- What material/system it applies to
- Why it is important
- Page number

## 2. Important Quantitative Results

Extract important numerical results such as:

- Strength
- Elongation
- Ductility
- Grain size
- Temperature
- Strain
- Strain rate
- Texture intensity
- Phase fraction
- Processing reduction
- Improvement percentage
- Other important numerical values

Create a table:

| Parameter | Value | Material / Condition | Significance | Page |
|---|---|---|---|---|

Only include numbers explicitly supported
by the paper context.

## 3. Processing → Microstructure Findings

Explain how processing affected:

- Grain structure
- Recrystallization
- Texture
- Second phases
- Precipitates
- Dislocations
- Twinning
- Other microstructural features

Use specific evidence from the paper.

## 4. Microstructure → Property Findings

Explain how microstructural changes affected:

- Strength
- Ductility
- Plasticity
- Formability
- Deformation behavior
- Other properties

Clearly explain the cause-and-effect relationship
when the paper supports it.

## 5. Deformation Mechanisms

Identify the important deformation mechanisms
discussed in the paper.

Examples include:

- Basal slip
- Non-basal slip
- Prismatic slip
- Pyramidal slip
- Twinning
- Dynamic recrystallization
- Grain boundary sliding
- Cross-slip
- Dislocation mechanisms

Do not assume a mechanism unless supported
by the paper.

## 6. Alloying / Composition Effects

Explain how changes in alloy composition
or alloying elements affected the observed
behavior.

Pay special attention to:

- Rare-earth elements
- Zn
- Sn
- Al
- Mn
- Ca
- Li
- Other alloying elements discussed

Only discuss elements actually present
in the paper.

## 7. Improvements Compared With Reference Conditions

Identify cases where the paper reports:

- Increased ductility
- Increased strength
- Improved formability
- Reduced anisotropy
- Modified texture
- Improved deformation behavior
- Other improvements

Where numerical comparisons are available,
show them explicitly.

## 8. Limitations / Contradictory Findings

Identify:

- Limitations
- Conflicting observations
- Trade-offs
- Conditions where an approach does not work
- Remaining uncertainties

Do not invent limitations.

## 9. Most Important Scientific Contribution

Explain in 1–3 paragraphs what the paper
contributes to the field.

Answer:

"What did this paper teach researchers
that was not previously clear?"

## 10. Executive Summary

Finish with exactly 5 concise bullet points
containing the most important takeaways.

Research paper context:

{context}
"""

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    answer = generate_answer(
        prompt
    )

    # --------------------------------------------------------
    # Prepare source information
    # --------------------------------------------------------

    finding_results = {
        "ids": [[
            chunk["id"]
            for chunk in selected_chunks
        ]],

        "documents": [[
            chunk["document"]
            for chunk in selected_chunks
        ]],

        "metadatas": [[
            chunk["metadata"]
            for chunk in selected_chunks
        ]]
    }

    return (
        answer,
        finding_results
    )

def detect_research_gaps(
    collection,
    source_name,
    max_chunks=20
):
    """
    Detect research gaps using analysis-specific
    semantic retrieval.
    """

    analysis_query = """
    research gaps limitations unresolved problems
    unanswered scientific questions
    limitations of current research
    limitations of experimental methods
    missing experimental evidence
    insufficient data
    contradictory findings
    inconsistent results
    conflicting observations
    unexplored alloy compositions
    unexplored rare earth elements
    magnesium alloy research opportunities
    plastic deformability challenges
    poor ductility formability limitations
    texture control challenges
    deformation mechanism uncertainties
    microstructure-property uncertainties
    processing limitations
    future research directions
    recommendations for future work
    areas requiring further investigation
    """

    context, results = get_analysis_context(
        collection,
        source_name,
        analysis_query,
        top_k=max_chunks
    )

    documents = results[
        "documents"
    ][0]

    if not documents:

        return (
            "No research-gap information was found "
            "for this paper.",
            results
        )

    prompt = f"""
You are an expert research assistant specializing
in materials science, metallurgy, and magnesium alloys.

Analyze the following research paper and identify
the important research gaps, limitations, unresolved
scientific questions, and opportunities for future
research.

Paper:
{source_name}

IMPORTANT RULES:

1. Use ONLY the supplied research-paper evidence.

2. Do not invent research gaps.

3. Clearly distinguish between:

   A. Explicit Research Gap
   A limitation, unanswered question, or future
   research direction explicitly stated by the
   authors.

   B. Evidence-Based Research Opportunity
   A research opportunity that can reasonably be
   identified from the supplied evidence, but is
   NOT explicitly stated by the authors.

4. Never present an inferred research opportunity
   as an author-stated research gap.

5. Mention page numbers whenever possible.

6. If a potential gap cannot be supported by the
   supplied evidence, do not include it.

7. Pay particular attention to:

   - magnesium alloy design
   - rare-earth alloying
   - alloy composition
   - processing methods
   - texture evolution
   - recrystallization
   - microstructure
   - deformation mechanisms
   - plastic deformability
   - ductility
   - formability
   - mechanical properties
   - experimental limitations

8. If the paper is a review article, distinguish
   gaps in the existing research field from
   limitations of the review itself.

Organize the answer using exactly these sections:

## 1. Explicit Research Gaps

List research gaps that are directly stated
or clearly identified by the authors.

For each gap include:

- Research gap
- Evidence
- Why it matters
- Page number

Clearly label each as:

**Author-Stated Gap**

---

## 2. Limitations of Existing Research

Identify limitations in current knowledge,
experimental approaches, processing methods,
materials, or available data.

For each limitation explain:

- What is missing or limited?
- Why does this limit understanding?
- What evidence supports this?

---

## 3. Unresolved Scientific Questions

Identify scientific questions that remain
unanswered.

Focus on questions involving:

- alloy composition
- rare-earth effects
- texture
- recrystallization
- grain structure
- deformation mechanisms
- plastic deformability
- processing-property relationships

Do not invent questions that are unrelated
to the supplied evidence.

---

## 4. Contradictory or Inconsistent Findings

Identify cases where the literature or studies
discussed in the paper report:

- conflicting results
- inconsistent trends
- different effects of the same alloying element
- different processing responses
- competing explanations

Explain why the contradiction matters.

---

## 5. Missing Experimental Evidence

Identify important evidence that appears to be
missing or insufficient.

Examples may include:

- insufficient characterization
- missing processing parameters
- limited mechanical testing
- insufficient microstructural analysis
- missing texture measurements
- lack of long-term or large-scale validation
- insufficient comparison between alloy systems

Only include points supported by the evidence.

---

## 6. Materials / Alloy Research Opportunities

Identify potential opportunities related to:

- new alloy compositions
- rare-earth combinations
- reduced-cost alloying strategies
- optimization of alloying concentration
- processing routes
- texture modification
- grain refinement
- improved plastic deformability

Clearly label these as:

**Evidence-Based Research Opportunity**

unless the authors explicitly state them.

---

## 7. Priority Research Gaps

Create a table:

| Priority | Research Gap | Evidence | Scientific Importance | Suggested Direction |

Use:

- High
- Medium
- Low

for priority.

Do not create priorities without evidence.

---

## 8. Potential Future Research Directions

Based ONLY on the identified gaps, suggest
reasonable future research directions.

For each direction explain:

- What should be investigated?
- Why?
- What material/system could be studied?
- What experimental or computational approach
  could address the gap?

Clearly distinguish suggestions from
author-stated future work.

---

## 9. Research Gap Summary

Finish with exactly 5 concise bullet points.

Each bullet should identify one of the most
important research gaps or future opportunities.

RESEARCH PAPER EVIDENCE:

{context}
"""

    answer = generate_answer(
        prompt
    )

    return (
        answer,
        results
    )
    """
    Identify research gaps, limitations,
    unresolved questions, contradictions,
    and potential future research directions
    from a selected research paper.
    """

    # --------------------------------------------------------
    # Get chunks belonging to selected paper
    # --------------------------------------------------------

    results = collection.get(
        where={
            "source": source_name
        },
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = results.get(
        "documents",
        []
    )

    metadatas = results.get(
        "metadatas",
        []
    )

    ids = results.get(
        "ids",
        []
    )

    # --------------------------------------------------------
    # No document found
    # --------------------------------------------------------

    if not documents:

        return (
            "No content was found for this paper.",
            {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]]
            }
        )

    # --------------------------------------------------------
    # Remove reference chunks
    # --------------------------------------------------------

    filtered_chunks = []

    for i in range(len(documents)):

        metadata = metadatas[i]

        if metadata.get(
            "is_reference",
            False
        ):
            continue

        text = documents[i]

        if looks_like_reference_chunk(text):
            continue

        filtered_chunks.append({
            "id": ids[i],
            "document": text,
            "metadata": metadata
        })

    # --------------------------------------------------------
    # Sort chunks by page
    # --------------------------------------------------------

    filtered_chunks.sort(
        key=lambda x: (
            x["metadata"].get(
                "page_number",
                0
            ),
            x["id"]
        )
    )

    # --------------------------------------------------------
    # Select representative chunks
    # --------------------------------------------------------

    if len(filtered_chunks) > max_chunks:

        step = (
            len(filtered_chunks)
            / max_chunks
        )

        selected_chunks = []

        for i in range(max_chunks):

            index = int(
                i * step
            )

            selected_chunks.append(
                filtered_chunks[index]
            )

    else:

        selected_chunks = filtered_chunks

    # --------------------------------------------------------
    # Build research context
    # --------------------------------------------------------

    context_parts = []

    for chunk in selected_chunks:

        page = chunk[
            "metadata"
        ].get(
            "page_number",
            "Unknown"
        )

        text = chunk[
            "document"
        ]

        context_parts.append(
            f"""
--- Page {page} ---

{text}
"""
        )

    context = "\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Prompt Gemini
    # --------------------------------------------------------

    prompt = f"""
You are an expert research analyst specializing
in materials science and engineering.

Analyze the following research paper and identify
its genuine research gaps.

Paper:
{source_name}

Use ONLY the supplied paper context.

Do not invent facts or claim that something is
a research gap unless the paper provides evidence
for that conclusion.

IMPORTANT:

Distinguish between:

1. Explicitly stated research gaps
2. Limitations explicitly mentioned by the authors
3. Unresolved questions identified in the paper
4. Contradictions or disagreements between studies
5. Reasonable research opportunities inferred
   from the evidence

Clearly label inferred opportunities as:

"Inferred research opportunity"

Do not present an inference as an explicit
statement from the authors.

Mention page numbers whenever possible.

Organize the answer as follows:

## 1. Explicit Research Gaps

Identify research gaps directly stated by
the authors.

For each gap provide:

- Research gap
- Evidence
- Page
- Why it matters

## 2. Limitations of Existing Research

Identify limitations discussed in the paper.

Consider:

- Experimental limitations
- Processing limitations
- Material limitations
- Characterization limitations
- Modeling limitations
- Dataset limitations
- Scale-up limitations

## 3. Unresolved Scientific Questions

Identify questions that remain unanswered.

For example:

- Why does a particular alloying element
  produce a certain texture?
- Why does a processing route improve ductility
  under one condition but not another?
- Which deformation mechanism dominates
  under particular conditions?

Only include questions supported by the paper.

## 4. Contradictory or Inconsistent Findings

Identify cases where:

- Different studies report different results
- Different alloy systems behave differently
- Processing conditions produce conflicting effects
- Proposed mechanisms disagree

Explain the contradiction and cite the
relevant pages.

## 5. Missing Experimental Evidence

Identify areas where additional experiments,
characterization, or validation could strengthen
the conclusions.

Examples:

- EBSD
- TEM
- XRD
- In-situ testing
- Mechanical testing
- Texture measurements
- Computational modeling

Do not recommend a technique unless it is
relevant to the identified gap.

## 6. Materials / Alloy Research Opportunities

Identify potential research opportunities
related to:

- Alloy composition
- Rare-earth additions
- Processing conditions
- Texture control
- Grain refinement
- Recrystallization
- Deformation mechanisms
- Formability
- Mechanical properties

Clearly label these as:

"Inferred research opportunity"

when they are not explicitly stated by the authors.

## 7. Priority Research Gaps

Rank the most important gaps.

Create a table:

| Priority | Research Gap | Evidence | Potential Impact | Page |
|---|---|---|---|---|

Use:

High
Medium
Low

for priority.

## 8. Potential Future Research Directions

Based strictly on the identified gaps,
suggest realistic research directions.

For each direction explain:

- Research question
- Why it matters
- Suggested approach
- Expected contribution

Clearly distinguish suggestions from
author-stated future work.

## 9. Research Gap Summary

Finish with exactly 5 concise bullets:

- Most important unresolved problem
- Most important experimental limitation
- Most important contradiction
- Most promising research opportunity
- Most important future direction

Research paper context:

{context}
"""

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    answer = generate_answer(
        prompt
    )

    # --------------------------------------------------------
    # Prepare source information
    # --------------------------------------------------------

    gap_results = {
        "ids": [[
            chunk["id"]
            for chunk in selected_chunks
        ]],

        "documents": [[
            chunk["document"]
            for chunk in selected_chunks
        ]],

        "metadatas": [[
            chunk["metadata"]
            for chunk in selected_chunks
        ]]
    }

    return (
        answer,
        gap_results
    )


def compare_papers(
    collection,
    paper_a,
    paper_b,
    max_chunks=20
):
    """
    Compare two research papers using
    comparison-specific semantic retrieval.
    """

    # ---------------------------------------------------------
    # Comparison-specific retrieval query
    # ---------------------------------------------------------

    comparison_query = """
    research paper comparison
    research objective research problem
    materials magnesium alloys alloy systems
    chemical composition alloying elements
    rare earth elements
    processing methods experimental approaches
    casting homogenization heat treatment
    extrusion rolling forging deformation processing
    processing parameters
    microstructure grain size recrystallization
    texture evolution basal texture
    deformation mechanisms slip twinning
    plastic deformability ductility formability
    mechanical properties strength elongation
    major scientific findings
    conclusions
    limitations
    research gaps
    future research
    similarities differences
    advantages disadvantages
    """

    # ---------------------------------------------------------
    # Retrieve comparison-specific evidence
    # separately from each paper
    # ---------------------------------------------------------

    context_a, results_a = get_analysis_context(
        collection,
        paper_a,
        comparison_query,
        top_k=max_chunks
    )

    context_b, results_b = get_analysis_context(
        collection,
        paper_b,
        comparison_query,
        top_k=max_chunks
    )

    documents_a = results_a[
        "documents"
    ][0]

    documents_b = results_b[
        "documents"
    ][0]

    # ---------------------------------------------------------
    # Check whether both papers contain usable evidence
    # ---------------------------------------------------------

    if not documents_a:

        return (
            f"No relevant comparison evidence was found "
            f"for Paper A: {paper_a}",
            {
                "paper_a": results_a,
                "paper_b": results_b
            }
        )

    if not documents_b:

        return (
            f"No relevant comparison evidence was found "
            f"for Paper B: {paper_b}",
            {
                "paper_a": results_a,
                "paper_b": results_b
            }
        )

    # ---------------------------------------------------------
    # Build comparison prompt
    # ---------------------------------------------------------

    prompt = f"""
You are an expert research assistant specializing
in materials science, metallurgy, and magnesium alloys.

Compare the following two research papers using
ONLY the supplied evidence.

==================================================
PAPER A
==================================================

{paper_a}

Evidence:

{context_a}


==================================================
PAPER B
==================================================

{paper_b}

Evidence:

{context_b}


==================================================
IMPORTANT RULES
==================================================

1. Use ONLY the supplied evidence from the two papers.

2. Do not invent information that is not present
   in the supplied evidence.

3. Keep Paper A and Paper B evidence separate
   when describing their individual findings.

4. Do not assume that two similar terms mean
   exactly the same scientific result.

5. If information is unavailable for one paper,
   explicitly write:

   "Not reported in the available paper context."

6. Mention page numbers whenever possible.

7. If the papers are review papers, remember that
   they may summarize findings from many different
   studies. Do not treat every finding as though
   it came from one experiment.

8. Clearly distinguish between:

   - Similarities
   - Differences
   - Contradictions
   - Complementary findings

9. When comparing quantitative values, include the
   relevant alloy/material and processing condition.

10. Do not claim that one paper is "better" overall
    unless there is clear evidence supporting that
    conclusion.

==================================================
STRUCTURE
==================================================

## 1. Research Objective

Explain:

- objective of Paper A
- objective of Paper B
- key research problem addressed by each

---

## 2. Materials / Alloy Systems

Create a comparison table:

| Aspect | Paper A | Paper B |
|---|---|---|
| Base Material | | |
| Alloy Systems | | |
| Alloying Elements | | |
| Rare-Earth Elements | | |
| Main Compositions | | |

---

## 3. Processing / Experimental Approaches

Compare:

- casting
- homogenization
- heat treatment
- extrusion
- rolling
- forging
- severe plastic deformation
- deformation conditions
- characterization methods

Use a table where appropriate.

---

## 4. Microstructure and Texture

Compare the papers with respect to:

- grain size
- grain refinement
- recrystallization
- precipitates
- second phases
- texture evolution
- basal texture
- texture weakening
- texture modification

Explain important similarities and differences.

---

## 5. Deformation Mechanisms

Compare the reported roles of:

- basal slip
- non-basal slip
- prismatic slip
- pyramidal slip
- twinning
- dislocation activity
- dynamic recrystallization
- grain-boundary mechanisms

Clearly identify where the papers agree
or disagree.

---

## 6. Mechanical Properties / Plastic Deformability

Compare reported effects on:

- yield strength
- tensile strength
- ductility
- elongation
- plastic deformability
- formability
- fracture behavior

Include numerical values only when supported
by the evidence.

---

## 7. Major Findings

Create two subsections:

### Paper A — Major Findings

List the most important findings.

### Paper B — Major Findings

List the most important findings.

---

## 8. Similarities

Identify the major scientific similarities
between the two papers.

Focus on:

- alloy design
- processing
- texture
- microstructure
- deformation mechanisms
- plastic deformability

---

## 9. Differences

Create a table:

| Scientific Aspect | Paper A | Paper B | Key Difference |
|---|---|---|---|

Focus on meaningful scientific differences,
not superficial differences.

---

## 10. Contradictions

Identify any findings that appear to conflict.

For each contradiction explain:

- what Paper A reports
- what Paper B reports
- possible reason for the difference
  ONLY if supported by the evidence

If there are no clear contradictions, state:

"No clear contradiction was identified in the
available paper evidence."

---

## 11. Research Gaps

Compare the research gaps identified
by both papers.

Identify:

- gaps shared by both papers
- gaps unique to Paper A
- gaps unique to Paper B
- unresolved questions common to the field

Do not invent gaps.

---

## 12. Which Paper Is More Useful For What?

Give a practical comparison.

For example:

- Paper A is more useful for understanding ______
- Paper B is more useful for understanding ______
- Both are useful for understanding ______

Base these statements strictly on the
supplied evidence.

---

## 13. Overall Comparison

Provide a concise scientific comparison
covering:

- objective
- materials
- processing
- microstructure
- texture
- deformation mechanisms
- properties
- major contribution

Explain how the two papers complement
each other.

---

## 14. Executive Comparison

Finish with exactly 5 concise bullet points.

Each bullet should communicate one important
comparison insight.

==================================================
FINAL REQUIREMENT
==================================================

Do NOT merge evidence from Paper A and Paper B
into a single unsupported claim.

Every major comparison should be traceable
to the supplied evidence.

PAPER A EVIDENCE:
{context_a}

PAPER B EVIDENCE:
{context_b}
"""

    # ---------------------------------------------------------
    # Generate comparison
    # ---------------------------------------------------------

    answer = generate_answer(
        prompt
    )

    # ---------------------------------------------------------
    # Return comparison + separate evidence
    # ---------------------------------------------------------

    return (
        answer,
        {
            "paper_a": results_a,
            "paper_b": results_b
        }
    )
    """
    Compare two research papers using
    paper-specific retrieved evidence.
    """

    # ========================================================
    # Helper function for retrieving one paper
    # ========================================================

    def get_paper_chunks(source_name):

        results = collection.get(
            where={
                "source": source_name
            },
            include=[
                "documents",
                "metadatas"
            ]
        )

        documents = results.get(
            "documents",
            []
        )

        metadatas = results.get(
            "metadatas",
            []
        )

        ids = results.get(
            "ids",
            []
        )

        filtered_chunks = []

        for i in range(len(documents)):

            metadata = metadatas[i]

            # -----------------------------------------------
            # Remove reference sections
            # -----------------------------------------------

            if metadata.get(
                "is_reference",
                False
            ):
                continue

            text = documents[i]

            if looks_like_reference_chunk(text):
                continue

            filtered_chunks.append({
                "id": ids[i],
                "document": text,
                "metadata": metadata
            })

        # ----------------------------------------------------
        # Sort by page
        # ----------------------------------------------------

        filtered_chunks.sort(
            key=lambda x: (
                x["metadata"].get(
                    "page_number",
                    0
                ),
                x["id"]
            )
        )

        # ----------------------------------------------------
        # Select representative chunks
        # ----------------------------------------------------

        if len(filtered_chunks) > max_chunks:

            step = (
                len(filtered_chunks)
                / max_chunks
            )

            selected_chunks = []

            for i in range(max_chunks):

                index = int(
                    i * step
                )

                selected_chunks.append(
                    filtered_chunks[index]
                )

        else:

            selected_chunks = filtered_chunks

        return selected_chunks


    # ========================================================
    # Retrieve Paper A
    # ========================================================

    chunks_a = get_paper_chunks(
        paper_a
    )


    # ========================================================
    # Retrieve Paper B
    # ========================================================

    chunks_b = get_paper_chunks(
        paper_b
    )


    # ========================================================
    # Check whether both papers exist
    # ========================================================

    if not chunks_a:

        return (
            f"No usable content was found for Paper A: "
            f"{paper_a}",

            {
                "paper_a": {
                    "ids": [],
                    "documents": [],
                    "metadatas": []
                },

                "paper_b": {
                    "ids": [],
                    "documents": [],
                    "metadatas": []
                }
            }
        )


    if not chunks_b:

        return (
            f"No usable content was found for Paper B: "
            f"{paper_b}",

            {
                "paper_a": {
                    "ids": [
                        chunk["id"]
                        for chunk in chunks_a
                    ],

                    "documents": [
                        chunk["document"]
                        for chunk in chunks_a
                    ],

                    "metadatas": [
                        chunk["metadata"]
                        for chunk in chunks_a
                    ]
                },

                "paper_b": {
                    "ids": [],
                    "documents": [],
                    "metadatas": []
                }
            }
        )


    # ========================================================
    # Build Paper A context
    # ========================================================

    context_a_parts = []

    for chunk in chunks_a:

        page = chunk[
            "metadata"
        ].get(
            "page_number",
            "Unknown"
        )

        text = chunk[
            "document"
        ]

        context_a_parts.append(
            f"""
--- Paper A | Page {page} ---

{text}
"""
        )

    context_a = "\n".join(
        context_a_parts
    )


    # ========================================================
    # Build Paper B context
    # ========================================================

    context_b_parts = []

    for chunk in chunks_b:

        page = chunk[
            "metadata"
        ].get(
            "page_number",
            "Unknown"
        )

        text = chunk[
            "document"
        ]

        context_b_parts.append(
            f"""
--- Paper B | Page {page} ---

{text}
"""
        )

    context_b = "\n".join(
        context_b_parts
    )


    # ========================================================
    # Comparison prompt
    # ========================================================

    prompt = f"""
You are an expert materials-science research
analyst.

Compare the following two research papers.

PAPER A:
{paper_a}

PAPER B:
{paper_b}

IMPORTANT RULES:

1. Use ONLY the supplied evidence.

2. Do not invent information.

3. Keep Paper A and Paper B evidence separate.

4. Clearly distinguish direct findings from
   interpretations.

5. Mention page numbers whenever possible.

6. If information is unavailable, write:
   "Not reported in the available paper context."

7. Do not assume that two papers use the same
   material, processing route, or mechanism unless
   the evidence supports that conclusion.

Analyze the papers using the following structure:

## 1. Research Objective

Compare the objectives and research questions
of both papers.

| Aspect | Paper A | Paper B |
|---|---|---|

## 2. Materials / Alloy Systems

Compare:

- Base materials
- Alloy systems
- Alloying elements
- Compositions
- Rare-earth additions
- Material conditions

| Aspect | Paper A | Paper B |
|---|---|---|

## 3. Processing / Experimental Approaches

Compare:

- Processing methods
- Heat treatment
- Deformation methods
- Experimental conditions
- Characterization techniques

| Aspect | Paper A | Paper B |
|---|---|---|

## 4. Microstructure and Texture

Compare:

- Grain structure
- Grain refinement
- Recrystallization
- Texture
- Second phases
- Precipitates
- Other microstructural features

Explain important similarities and differences.

## 5. Deformation Mechanisms

Compare the mechanisms discussed by both papers.

Consider:

- Basal slip
- Non-basal slip
- Prismatic slip
- Pyramidal slip
- Twinning
- Dynamic recrystallization
- Dislocation mechanisms
- Other mechanisms

Only include mechanisms supported by the evidence.

## 6. Mechanical Properties / Deformability

Compare:

- Strength
- Ductility
- Elongation
- Formability
- Plastic deformability
- Other relevant properties

Include numerical values when explicitly reported.

## 7. Major Findings

Create a table:

| Finding | Paper A | Paper B |
|---|---|---|

Explain the most important finding from each paper.

## 8. Similarities

Identify the major conclusions or mechanisms
where both papers agree.

## 9. Differences

Identify the most important differences
between the papers.

Explain possible reasons only when supported
by the evidence.

## 10. Contradictions

Identify any cases where the papers report
different or apparently contradictory findings.

Explain the evidence behind each difference.

## 11. Research Gaps

Identify:

- What Paper A leaves unanswered
- What Paper B leaves unanswered
- Questions neither paper completely addresses

## 12. Which Paper Is More Useful For What?

Do NOT simply declare one paper "better."

Instead explain:

- Which paper is more useful for understanding
  materials/composition?
- Which is more useful for processing?
- Which is more useful for deformation mechanisms?
- Which is more useful for practical applications?
- Which provides stronger experimental evidence?

## 13. Overall Comparison

Give a concise overall assessment explaining
how the two papers complement each other.

## 14. Executive Comparison

Finish with exactly 5 bullets summarizing
the most important differences and similarities.

-----------------------------
PAPER A EVIDENCE
-----------------------------

{context_a}

-----------------------------
PAPER B EVIDENCE
-----------------------------

{context_b}
"""

    # ========================================================
    # Generate comparison
    # ========================================================

    answer = generate_answer(
        prompt
    )


    # ========================================================
    # Prepare evidence for UI
    # ========================================================

    comparison_results = {

        "paper_a": {
            "ids": [
                chunk["id"]
                for chunk in chunks_a
            ],

            "documents": [
                chunk["document"]
                for chunk in chunks_a
            ],

            "metadatas": [
                chunk["metadata"]
                for chunk in chunks_a
            ]
        },

        "paper_b": {
            "ids": [
                chunk["id"]
                for chunk in chunks_b
            ],

            "documents": [
                chunk["document"]
                for chunk in chunks_b
            ],

            "metadatas": [
                chunk["metadata"]
                for chunk in chunks_b
            ]
        }
    }


    return (
        answer,
        comparison_results
    )

def get_analysis_context(
    collection,
    source_name,
    analysis_query,
    top_k=20
):
    """
    Retrieve high-quality analysis-specific evidence
    using hybrid semantic + BM25 retrieval followed
    by CrossEncoder reranking.
    """

    results = search_hybrid_reranked(
        collection=collection,
        query=analysis_query,
        source_name=source_name,
        top_k=top_k,
        candidate_k=max(
            top_k * 2,
            20
        ),
        semantic_weight=0.6,
        bm25_weight=0.4
    )

    documents = results[
        "documents"
    ][0]

    metadatas = results[
        "metadatas"
    ][0]

    context_parts = []

    for i in range(
        len(documents)
    ):

        page = metadatas[i].get(
            "page_number",
            "Unknown"
        )

        text = documents[i]

        context_parts.append(
            f"""
--- Page {page} ---

{text}
"""
        )

    context = "\n".join(
        context_parts
    )

    return (
        context,
        results
    )
    """
    Retrieve analysis-specific evidence from
    one selected research paper.
    """

    results = search_collection_for_document(
        collection=collection,
        query=analysis_query,
        source_name=source_name,
        top_k=top_k
    )

    documents = results[
        "documents"
    ][0]

    metadatas = results[
        "metadatas"
    ][0]

    ids = results[
        "ids"
    ][0]

    context_parts = []

    for i in range(
        len(documents)
    ):

        page = metadatas[i].get(
            "page_number",
            "Unknown"
        )

        text = documents[i]

        context_parts.append(
            f"""
--- Page {page} ---

{text}
"""
        )

    context = "\n".join(
        context_parts
    )

    return (
        context,
        results
    )