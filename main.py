import pdfplumber
from transformers import pipeline

# Step 1: Extract text from the PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Step 2: Extract alteration clauses
def extract_alteration_clauses(text):
    clauses = text.split('\n')
    alteration_clauses = [clause for clause in clauses if "alteration" in clause.lower()]
    return alteration_clauses

# Step 3: Process alteration clauses to extract details
def extract_alteration_details(clauses):
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-large-squad2")
    extracted_details = []

    for clause in clauses:
        details = {
            "Consent (Y/N)": None,
            "Cost": None,
            "Structural or Non-structural": None,
            "Plan Submission to LL": None,
            "LL Submission Fee": None,
            "Summary of Alteration": None,
        }

        try:
            # Ask specific questions about the clause
            details["Consent (Y/N)"] = qa_pipeline(
                question="Does this clause mention consent as yes or no?", context=clause
            )["answer"]
            details["Cost"] = qa_pipeline(
                question="What is the cost mentioned in this clause?", context=clause
            )["answer"]
            details["Structural or Non-structural"] = qa_pipeline(
                question="Is the alteration structural or non-structural?", context=clause
            )["answer"]
            details["Plan Submission to LL"] = qa_pipeline(
                question="Is there a mention of plan submission to LL?", context=clause
            )["answer"]
            details["LL Submission Fee"] = qa_pipeline(
                question="What is the LL submission fee mentioned?", context=clause
            )["answer"]
            details["Summary of Alteration"] = clause
        except Exception as e:
            # Handle errors gracefully
            details["Error"] = str(e)

        extracted_details.append(details)

    return extracted_details

# Step 4: Summarize the text
def summarize_text_llm(text, max_length=130, min_length=30):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text[:3000], max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Step 5: Extract specific data using a pre-trained question-answering model
def extract_data_with_llm(context, questions):
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-large-squad2")
    extracted_data = {}
    for field, question in questions.items():
        try:
            answer = qa_pipeline(question=question, context=context)
            extracted_data[field] = answer['answer']
        except Exception as e:
            extracted_data[field] = None
    return extracted_data

# Main function to process the PDF
def process_pdf_with_llm(pdf_path):
    # Step 1: Extract text
    print("Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(pdf_path)
    print("Text extraction complete.")

    # Step 2: Extract alteration clauses
    print("Extracting alteration clauses...")
    alteration_clauses = extract_alteration_clauses(pdf_text)
    print(f"Found {len(alteration_clauses)} alteration clauses.")

    # Step 3: Extract alteration details
    print("Extracting alteration details...")
    alteration_details = extract_alteration_details(alteration_clauses)
    print("Alteration details extraction complete.")

    # Step 4: Summarize the text
    print("Summarizing text with LLM...")
    summary = summarize_text_llm(pdf_text)
    print("Summarization complete.")

    # Step 5: Extract required data
    print("Extracting data with LLM...")
    questions = {
        "Property's Address 1": "What is the primary address of the property?",
        "Ownership Type": "What is the ownership type?",
        "Client Name": "Who is the client mentioned in the document?",
        "Property's Address 2": "Is there a secondary address mentioned for the property?",
        "Property Status": "What is the status of the property?",
        "Property Name": "What is the name of the property?",
        "Property's City": "In which city is the property located?",
        "Region": "What is the region of the property?"
    }
    extracted_data = extract_data_with_llm(pdf_text, questions)
    print("Data extraction complete.")

    # Combine results
    result = {
        "Extracted Data": extracted_data,
        "Summary": summary,
        "Alteration Details": alteration_details
    }
    return result

# Entry point
if __name__ == "__main__":
    # Path to the uploaded PDF file
    pdf_path = "6001851_Tully ACC  CL697_Fully execd lease_01.03.1428.02.17.pdf"

    # Process the PDF and generate the JSON
    result_json = process_pdf_with_llm(pdf_path)

    # Output the results
    import json
    print("\n--- Result JSON ---")
    print(json.dumps(result_json, indent=4))
