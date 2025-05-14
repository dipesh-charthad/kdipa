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
    "I shall only use the given dataset as my sole source of truth. I will not fabricate, infer, or assume any data beyond what is explicitly available. If sufficient data is not available, I will state so clearly without making assumptions. I shall **not add or invent any new parameters** within the final output. I will only provide the following response fields exactly as listed: **Decision**, **DecisionExplanation**, **Recommendations**, and **Risks Identified**—with no additional fields."
 
    Given the new application details:
    {application_data}
 
    And the indexed dataset:
    {indexed_dataset}
 
    You must evaluate and compare applications **only using the following essential fields**:
    - "companyName"
    - "companyOrigin"
    - "companyCity"
    - "companyOutput"
    - "contributionAmount"
    - "totalCapitalAmount"
    - "shareholderCompanyPartnerName"
    - "shareholderNationality"
    - "valueOfEquityOrShares"
    - "percentageOfEquityOrShares"
    - "numberOfEquityOrShares"
    - "totalInvestmentValue"
    **Note**: Do not use the following fields for comparison:
    - "licenseType"
    - "appType"
    - "contributionType"
    - "termsAndConditions"
 
    Provide the following information:
 
    **Top 3 Similar Applications** – Identify the **top 3 applications** that are most similar to the new application from the indexed dataset where appState == ACCEPTED or appState == REJECTED.
    - If fewer than 3 similar applications exist, fill in the remaining entries with the **next closest possible matches**, even if the similarity is weak. The output must always include exactly 3 similar applications.
    - If no similar applications are found, return the **nearest matches** instead of an empty list.
 
    For each similar application, present the details in the following structure:
    - **UUID** (Unique identifier of the application)
    - **Percentage Matching** (How closely it matches the new application in XX% format)
    - **Description** (Summarized key details concisely in 2-3 sentences)
    - **Status** (ACCEPTED/REJECTED)
 
    **Decision:** Based solely on historical approval/rejection trends within the dataset and using only the essential fields, decide whether to **ACCEPTED** or **REJECTED** the application.
 
    **DecisionExplanation:** Always provide a clear, data-backed explanation for the decision, regardless of whether the outcome is ACCEPTED or REJECTED.
    - If insufficient data is available to determine a confident decision, explicitly state so.
 
    **Recommendations:** Suggest changes that could improve the likelihood of approval, based only on patterns and correlations in the essential fields from historical data.
    - Always format the output as bullet points, even if there is only one recommendation.
    - Do not offer speculative or fabricated recommendations.
 
    **Risks Identified:** Identify any potential risks associated with this application by analyzing patterns in the essential fields of historically **rejected** applications.
    - Always format the output as bullet points, even if there is only one risk or no known risks.
    - Mention specific risk factors if similar rejected applications exhibited them.
    - If no known risks are found based on the essential fields, explicitly say so using a bullet point.

    All output should be formatted in **JSON**.
    """
    return prompt