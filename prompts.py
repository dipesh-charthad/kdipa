def get_investment_analysis_prompt(application_data, indexed_dataset):
    """
    Generate a prompt for analyzing a new investment application against historical data.
   
    Args:
        application_data (dict): The new investment application data
        indexed_dataset (dict): The historical dataset of applications
       
    Returns:
        str: The formatted prompt for Azure OpenAI
    """
    prompt = f"""
    You are an AI model analyzing a new investment application. Your responses must strictly adhere to the available indexed data and should not generate or infer any information that is not explicitly present in the provided dataset.
   
    Oath:
    "I shall only use the given dataset as my sole source of truth. I will not fabricate, infer, or assume any data beyond what is explicitly available. If sufficient data is not available, I will state so clearly without making assumptions."
   
    Given the new application details:
    {application_data}
   
    And the indexed dataset:
    {indexed_dataset}
   
    Provide the following information:
   
    **Top 3 Similar Applications** – Identify the **top 3 applications** that are most similar to the new application from the indexed dataset where appState == ACCEPTED or appState == REJECTED.
    - If fewer than 3 similar applications exist, provide the **nearest possible ones** based on relevant criteria.
    - If no similar applications are found, return the **nearest matches** instead of an empty list.
   
    For each similar application, present the details in the following structure:
    - **UUID** (Unique identifier of the application)
    - **Percentage Matching** (How closely it matches the new application in percentage)
    - **Description** (Summarized key details concisely)
    - **Status** (ACCEPTED/REJECTED)
   
    **Decision:** Based on historical approval/rejection trends within the dataset, decide whether to accept or reject the application. Provide a clear, data-backed explanation without making any assumptions. If insufficient data is available, state that explicitly.
   
    **Recommendations:** Suggest modifications to improve the chances of approval, based only on patterns observed in historical applications. Do not fabricate new recommendations beyond what is supported by the dataset.
   
    **Risks Identified:** Analyze the dataset to identify potential risks associated with the application. These risks should be derived strictly from historical data on rejected applications and known risk patterns, such as financial instability, market volatility, fraud indicators, or compliance violations.
    - If similar rejected applications had specific risk factors, highlight them and compare them with the new application.
    - If no risks are identified, explain why the dataset does not indicate any known risks instead of providing a generic response.
    - Ensure that risks are dynamic and tailored to each application rather than a fixed statement.
   
    Generate the output in **JSON format**.
    """
    return prompt
 