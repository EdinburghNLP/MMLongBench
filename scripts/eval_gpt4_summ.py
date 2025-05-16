import argparse
import json
import os
import sys
import re
from tqdm import tqdm
import time
import glob
from functools import partial

import numpy as np

import openai

def call_api(func, limit: int=5, pause: int=10, return_rate_limit=False):
    """
    Call the API function with retries and rate limit handling.
    TODO: more error handling?
    """
    count = 0
    while True:
        try:
            output = func()
            break
        except Exception as e:
            print(f"Exception while using api: {e}")
            msg = str(e).lower()
            if "rate limit" in msg or "rate_limit" in msg or "quota" in msg or "429" in msg or ("overloaded" in msg and count >= limit):
                if return_rate_limit:
                    print(f"Rate limit exceeded, returning")
                    return "rate limit"
                else:
                    print(f"Rate limit exceeded, waiting {pause} secs and retrying...")
            if count < limit:
                print(f"Encountered error {e}, retrying...")
                count += 1
                time.sleep(pause)
            else:
                print("Skipping generation due to unknown error")
                raise e
    return output


# prompts inspired by https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG
fluency_prompt_gov="""Please act as an impartial judge and evaluate the fluency of the provided text. The text should be coherent, non-repetitive, fluent, and grammatically correct.

Below is your grading rubric:
- Score 0 (incoherent, repetitive, or incomplete): Incoherent sentences, repetitive sentences (even if not by exact words), incomplete answers, or gibberish. Note that even if the answer is coherent, if it is repetitive or incomplete, it should be given a score of 0.
  - Examples:
    - Incomplete: "Summary:"
    - Incoherent: "Summary:U.S. agencies engaged export and controls controls controls controls diversion prevent items U.S. activities compliance allies transshipment risk misuse exported misuse misuse illicit illicit against interests or."
    - Repetitive: "Summary:The audit focused on determining the cost and schedule performance of selected programs. The audit focused on determining the cost and schedule performance of selected programs. The audit focused on determining the cost and schedule performance of selected programs. The audit focused on determining the cost and schedule performance of selected programs."

- Score 1 (coherent, non-repetitive answer): Coherent, non-repetitive, fluent, grammatically correct answers. If the text is coherent, non-repetitive, and fluent, but the last sentence is truncated, it should still be given a score of 1.
  - Examples:
    - "Why GAO Did This Study:\nTobacco use is the leading cause of preventable death and disease in the United States. In 2009, the Family Smoking Prevention and Tobacco Control Act (Tobacco Control Act) granted FDA, an agency within the Department of Health and Human Services (HHS), authority to regulate tobacco products, including marketing and distribution to youth. The act established CTP, which implements the act by educating the public on the dangers of tobacco use; developing the science needed for tobacco regulation; and developing and enforcing regulations on the manufacture, marketing, and distribution of tobacco products. The act authorized FDA to assess and collect user fees from tobacco manufacturers and importers.\nThe Tobacco Control Act mandated that GAO review the authority and resources provided to FDA for regulating the manufacture, marketing, and distribution of tobacco products. This report examines (1) how FDA spent tobacco user fees for key activities using its authorities granted in the act, and (2) any challenges FDA encountered in using its authorities. GAO analyzed data on tobacco user fees collected and spent on key activities by FDA as of March 31, 2014; reviewed documents related to FDA's key activities, as well as relevant laws, regulations, and guidance; and interviewed CTP, public health, and tobacco industry officials.\nHHS reviewed a draft of this report and agreed with GAO's reiteration of its previous recommendation that performance measures for all tobacco product reviews are needed.\n\nWhat GAO Found:\nAs of March 31, 2014, the Food and Drug Administration (FDA) spent about $1.48 billion (79 percent) of the $1.88 billion in total tobacco user fees it collected since fiscal year 2009. FDA spent the majority of tobacco user fees on key activities led by the agency's Center for Tobacco Products (CTP), which is funded solely by tobacco user fees. These included activities related to public education (including public education campaigns and communicating CTP activities); regulatory science (including research, product review, and developing the science to support regulations and guidance); and compliance and enforcement (including tobacco retailer inspections; manufacturer and import inspections and enforcement; promotion, advertising, and labeling surveillance; and outreach and small business assistance).\nWhile FDA has taken steps to address some of the challenges it has faced, including challenges related to starting up a new center, it continues to face challenges, including setting and monitoring review time frames. Until recently, CTP has not had performance measures for making final decisions on new tobacco product submissions by which to assess its progress, as GAO previously recommended. FDA has announced performance measures for two of its new tobacco product review processes (to take effect in October 2014), but not for the type of new tobacco product submission that comprises the bulk of FDA's review backlog. The agency has indicated that it intends to establish such performance measures, but until it does so, the agency's ability to assess its efforts will be limited. This will be particularly pressing as FDA moves forward with plans to deem additional types of tobacco products to be subject to"

Now, read the provided text, and evaluate the fluency using the rubric. Then output your score in the following json format: {{"fluency": 1}}.

Text: "{text}"
"""

fluency_prompt_lexsum="""Please act as an impartial judge and evaluate the fluency of the provided text. The text should be coherent, non-repetitive, fluent, and grammatically correct.

Below is your grading rubric:
- Score 0 (incoherent, repetitive, or incomplete): Incoherent sentences, repetitive sentences (even if not by exact words), incomplete answers, or gibberish. Note that even if the answer is coherent, if it is repetitive or incomplete, it should be given a score of 0.
  - Examples:
    - Incomplete: "Summary:"
    - Incoherent: "Summary: The plaintiff the the the the able the the the the the the the the the the able the the the the the Ã�\n"
    - Repetitive: "Summary: The U.S. government brought a criminal case against four defendants. Summary: The U.S. government brought a criminal case against four defendants. Summary: The U.S. government brought a criminal case against four defendants. Summary: The U.S. government brought a criminal case against four defendants."

- Score 1 (coherent, non-repetitive answer): Coherent, non-repetitive, fluent, grammatically correct answers. If the text is coherent, non-repetitive, and fluent, but the last sentence is truncated, it should still be given a score of 1.
  - Examples:
    - "This case is about an apprenticeship test that had a disparate impact on Black apprenticeship applicants. The Equal Employment Opportunity Commission (EEOC) filed this lawsuit on December 27, 2004, in U.S. District Court for the Southern District of Ohio."
    - "The plaintiffs sought declaratory and injunctive relief, as well as attorneys' fees and costs, under the Americans with Disabilities Act, the Rehabilitation Act of 1973, the Social Security Act, and the Nursing Home Reform Act. The case was certified as a class action on behalf of all Medicaid-eligible adults with disabilities in Cook County, Illinois, who are being, or may in the future be, unnecessarily confined to nursing facilities and with appropriate supports and services may be able to live in a community setting. The defendants denied the allegations and argued that the plaintiffs' claims were not typical of the class and that the class definition was too broad. The case is ongoing, with discovery and expert testimony scheduled for the fall of"

Now, read the provided text, and evaluate the fluency using the rubric. Then output your score in the following json format: {{"fluency": 1}}.

Text: "{text}"
"""

recall_prompt_gov="""Please act as an impartial judge and evaluate the quality of the provided summary of a government report from U.S. Government Accountability Office (GAO). The summary should discuss one or more of the following: why GAO did this study, what GAO found, and what GAO recommends. The text should contain all the major points in the expert-written summary, which are given to you.

Below is your grading rubric:
Recall:
- Evaluate the provided summary by deciding if each of the key points is present in the provided summary. A key point is considered present if its factual information is mostly-supported by the provided summary. If a key point contains multiple facts, it's still considered supported if most of the facts are present.
- Score: the number of key points mostly-supported by the provided summary.
- Examples: use the following examples to guide your evaluation.

Example 1:

Key points:
1. The Future Combat System (FCS) program is the centerpiece of the Army's effort to transition to a lighter combat force.
2. The FCS program is the centerpiece of the Army's effort to transition to a more agile combat force.
3. The FCS program is the centerpiece of the Army's effort to transition to a more capable combat force.
4. By law, GAO is to report annually on the FCS program.
5. Law requires the Department of Defense (DOD) to hold a milestone review of the FCS program.
6. This milestone review is now planned for 2009.
7. This report addresses (1) what knowledge will likely be available in key areas for the review.
8. This report addresses (2) the challenges that lie ahead following the review.
9. To meet these objectives, GAO reviewed key documents and performed analysis.
10. GAO attended demonstrations and design reviews to meet these objectives.
11. GAO interviewed DOD officials to meet these objectives.
12. The Army will be challenged to demonstrate the knowledge needed to warrant an unqualified commitment to the FCS program.
13. This challenge will occur at the 2009 milestone review.
14. The Army has made progress.
15. Knowledge deficiencies remain in key areas.
16. Specifically, all critical technologies are not currently at a minimum acceptable level of maturity.
17. It has not been demonstrated that emerging FCS system designs can meet specific requirements.
18. It has not been demonstrated that emerging FCS system designs can mitigate associated technical risks.
19. Actual demonstrations of FCS hardware and software—versus modeling and simulation results—have been limited.
20. Only small-scale warfighting concepts and limited prototypes have been demonstrated.
21. Network performance is also largely unproven.
22. These deficiencies do not necessarily represent problems that could have been avoided.
23. Rather, these deficiencies reflect the actual immaturity of the program.
24. Finally, there is an existing tension between program costs and available funds that seems only likely to worsen.
25. FCS costs are likely to increase at the same time as competition for funds intensifies between near- and far-term needs in DOD.
26. Competition for funds will also intensify between DOD and other federal agencies.
27. DOD could have at least three programmatic directions to consider for shaping investments in future capabilities.
28. Each of these programmatic directions presents challenges.
29. First, the current FCS acquisition strategy is unlikely to be executed within the current $159 billion cost estimate.
30. The strategy calls for significant production commitments before designs are demonstrated.
31. To date, FCS has spent about 60 percent of its development funds.
32. The most expensive activities remain to be done before the production decision.
33. In February 2010, Congress will be asked to begin advancing procurement funds for FCS core systems.
34. This request will happen before most prototype deliveries have taken place.
35. This request will happen before critical design review has taken place.
36. This request will happen before key system tests have taken place.
37. By the 2013 production decision, Congress will have been asked for over $50 billion in funding for FCS.
38. The program to spin out early FCS capabilities to current forces operates on an aggressive schedule centered on a 2009 demonstration.
39. The 2009 demonstration will employ some surrogate systems and preliminary designs instead of fully developed items.
40. There will be little time for evaluation of results.
41. Third, the Army is currently considering an incremental FCS strategy.
42. This strategy is to develop and field capabilities in stages versus in one step.
43. Such an approach is generally preferable.
44. This approach would present decision makers with a third major change in FCS strategy to consider anew.
45. While details are yet unavailable, it is important that each increment be justified by itself.
46. Each increment should not be dependent on future increments.

Summary: <start of summary>Why GAO Did This Study:
The Future Combat System (FCS) program is the centerpiece of the Army's effort to transition to a lighter combat force. By law, GAO is to report annually on the FCS program. This report addresses (1) what knowledge will likely be available in key areas for the review, and (2) the challenges that lie ahead following the review. To meet these objectives, GAO reviewed key documents and interviewed DOD officials.

What GAO Found:
The Army will be challenged to demonstrate the knowledge needed to warrant an unqualified commitment to the FCS program. While the Army has made progress, knowledge deficiencies remain in key areas. Specifically, all critical technologies are not currently at a minimum acceptable level of maturity. Actual demonstrations of FCS hardware and software have been limited. Network performance is also largely unproven. DOD could have at least three programmatic directions to consider for shaping investments in future capabilities. First, the current FCS acquisition strategy is unlikely to be executed within the current $159 billion cost estimate. To date, FCS has spent about 60 percent of its development funds. In February 2010, Congress will be asked to begin advancing procurement funds for FCS core systems before most prototype deliveries and key system tests have taken place. Second, the program to spin out early FCS capabilities to current forces operates on an aggressive schedule centered on a 2009 demonstration. Third, the Army is currently considering an incremental FCS strategy to develop and field capabilities in stages versus in one step. Such an approach is generally preferable.<end of summary>

Reasoning: The summary covers: FCS as Army's transition centerpiece (point 1), GAO's reporting requirement (point 4), report objectives (points 7, 8), GAO's methods (points 9, 11), Army's challenges (point 12), progress and deficiencies (points 14, 15), technology issues (points 16, 19, 21), three programmatic directions (points 27, 29, 31, 33, 34, 36, 38, 41-43). It omits: "more agile/capable" (points 2, 3), 2009 milestone review (points 5, 6, 13), demonstrations attendance (point 10), design requirements issues (points 17, 18), small-scale concepts (point 20), program immaturity explanation (points 22, 23), funding competition (points 24-26), challenges after review (point 28), production before design demonstration (points 30, 32), technology testing issues (point 35), $50 billion funding (point 37), surrogate systems (points 39, 40), and increment justification (points 44-46). The summary supports 22 key points.

Output: {{"supported_key_points": [1, 4, 7, 8, 9, 11, 12, 14, 15, 16, 19, 21, 27, 29, 31, 33, 34, 36, 38, 41, 42, 43], "recall": 22}}

Now, read the provided summary and key points, and evaluate the summary using the rubric. First, think step-by-step and provide your reasoning and assessment on the answer. Please keep your response concise and limited to a single paragraph. Then output your score in the following json format: {{"supported_key_points": [1, 4, 7, 8, 9, 11, 12, 14, 15, 16, 19, 21, 27, 29, 31, 33, 34, 36, 38, 41, 42, 43], "recall": 22}}, where "supported_key_points" contains the key points that are present in the summary and "recall" is the total number of key points present in the summary.

Key points:
{keypoints}

Summary: <start of summary>{summary}<end of summary>
"""
# "gov_report_gao-09-288"


recall_prompt_lexsum="""Please act as an impartial judge and evaluate the quality of the provided summary of a civil lawsuit. The summary is based on a set of legal documents, and it should contain a short description of the background, the parties involved, and the outcomes of the case. The text should contain all the major points in the expert-written summary, which are given to you.

Below is your grading rubric:
Recall:
- Evaluate the provided summary by deciding if each of the key points is present in the provided summary. A key point is considered present if its factual information is well-supported by the provided summary.
- Score: the number of key points present in the provided summary.
- Examples: use the following examples to guide your evaluation.

Example 1:

Key points:
1. The case challenged curfews in Los Angeles and San Bernardino, California.
2. The curfews were issued in response to the nationwide protests following the police killing of George Floyd in Minneapolis.
3. The complaint argued that the curfews violated free speech, free assembly, free movement, and Due Process.
4. The complaint also argued that the San Bernardino curfew violated the Establishment Clause.
5. The complaint sought injunctive and declaratory relief.
6. The plaintiffs voluntarily dismissed the case on July 7, 2020.
7. The dismissal occurred because the city had rescinded the curfews and not attempted to reinstate them.

Summary: "In June 2020, Black Lives Matter - Los Angeles and several individuals filed a lawsuit in the U.S. District Court for the Central District of California against Los Angeles Mayor Eric Garcetti, other city officials, and the City of San Bernardino, challenging the constitutionality of curfew orders imposed during protests against police violence. The plaintiffs, represented by the ACLU of Southern California, argued that the curfews violated their First Amendment rights to free speech and assembly, as well as their freedom of movement, by suppressing political protests and other activities. The lawsuit also claimed that the curfews were not narrowly tailored to address any emergency and lacked sufficient notice. However, the plaintiffs voluntarily dismissed the case in July 2020 after the defendants lifted the curfew orders and did not reinstate them in the following weeks."

Reasoning: The summary states that the plaintiffs challenged the constitutionality of curfew orders against Los Angeles and San Bernadino, so key point 1 is present. The summary does not mention that the curfew orders were issued in response to the nationwide protest that resulted from the police killing of George Floyd in Minneapolis, so key point 2 is missing. The summary does mention that the complaint argued that the curfews violated the First Amendment rights to free speech and assembly, so key point 3 is present. The summary does not mention that the complaint argued that the San Bernardino curfew violated the Establishment Clause, so key point 4 is missing. The summary does not mention that the complaint sought injunctive and declaratory relief, so key point 5 is missing. The summary mentions that the plaintiffs voluntarily dismissed the case in July 2020 after the defendants lifted the curfew orders and did not reinstate them in the following weeks, so key point 6 and 7 are present. Finally, key points 1, 3, 6, and 7 are present in the summary, so the recall score is 4.

Output: {{"recall": 4}}


Example 2:

Key points:
1. Individuals with disabilities brought the case against various Illinois state officials.
2. The plaintiffs sought declaratory and injunctive relief, alleging inappropriate institutionalization when community-based care was possible.
3. In August 2011, a consent decree was entered, requiring the state to transition class members from nursing facilities to community-based settings.
4. The transition plan was updated in April 2018.
5. Monitoring of the transition is ongoing as of November 2018.

Summary: "Five Medicaid-eligible individuals with disabilities, Lenil Colbert, Constance Gray, Ernest Reeves, Kenya Lyles, and Dwight Scott, filed a class action lawsuit in the United States District Court for the Northern District of Illinois against Illinois state officials, including Governor Rod R. Blagojevich, Secretary of the Illinois Department of Human Services Carol L. Adams, Director of the Illinois Department of Healthcare and Family Services Barry S. Maram, and Director of the Illinois Department of Public Health Eric E. Whitaker. The plaintiffs alleged that the defendants' policies and practices effectively compel people with disabilities to enter nursing facilities in order to receive long-term care and assistance, forcing them to forego liberty, privacy, independence, and the opportunity to live in the communities of their choice. The plaintiffs sought declaratory and injunctive relief, as well as attorneys' fees and costs, under the Americans with Disabilities Act, the Rehabilitation Act of 1973, the Social Security Act, and the Nursing Home Reform Act. The case was certified as a class action on behalf of all Medicaid-eligible adults with disabilities in Cook County, Illinois, who are being, or may in the future be, unnecessarily confined to nursing facilities and with appropriate supports and services may be able to live in a community setting. The defendants denied the allegations and argued that the plaintiffs' claims were not typical of the class and that the class definition was too broad. The case is ongoing, with discovery and expert testimony scheduled for the fall of"

Reasoning: The summary states that the plaintiffs brought the case against various Illinois state officials, so key point 1 is present. The summary mentions that "the plaintiffs sought declaratory and injunctive relief" and the practices "compelled people with disabilities to enter nursing facilities... to forego ... the opportunity to live in the communities of their choice", so key point 2 is present. The summary does not mention that a consent decree was entered in August 2011, so key point 3 is missing. The summary does not mention that the transition plan was updated in April 2018, so key point 4 is missing. The summary does not mention that monitoring of the transition is ongoing as of November 2018, so key point 5 is missing. Therefore, key points 1 and 2 are present so the recall score is 2.

Output: {{"recall": 2}}

Now, read the provided summary and key points, and evaluate the summary using the rubric. First, think step-by-step and provide your reasoning and assessment on the answer. Please keep your response concise and limited to a single paragraph. Then output your score in the following json format: {{"recall": 2}}.

Key points:
{keypoints}

Summary: "{summary}"
"""

precision_prompt_gov="""Please act as an impartial judge and evaluate the quality of the provided summary of a government report from U.S. Government Accountability Office (GAO). The summary should discuss one or more of the following: why GAO did this study, what GAO found, and what GAO recommends.

Below is your grading rubric:
Precision:
- Evaluate the provided summary by deciding if each sentence in the provided summary is supported by the information provided in the expert summary. A sentence is considered supported if its major facts align with the information in the expert summary. A sentence is still considered supported even if some of its minor details, such as dates, entity names, or the location, are not explicitly mentioned in the expert summary. A sentence is not supported if its major facts are not mentioned or contradicted in the expert summary. It is also not supported if it introduces new information not present in the expert summary, such as additional analysis or commentary on the story.
- Score: the number of sentences in the provided summary that are supported by the expert summary.
- Examples: use the following examples to guide your evaluation.

Example 1:

Expert summary: <start of summary>Why GAO Did This Study:
The Congressional Budget Office projects that federal deficits will reach $1 trillion in 2020 and average $1.2 trillion per year through 2029, further adding to the more than $16 trillion in current debt held by the public. As a result, Treasury will need to issue a substantial amount of debt to finance government operations and refinance maturing debt. To support its goal to borrow at the lowest cost over time, Treasury must maintain strong demand from a diverse group of investors for Treasury securities.
GAO prepared this report as part of continuing efforts to assist Congress in identifying and addressing debt management challenges. This report (1) identifies factors that affect demand for Treasury securities and (2) examines how Treasury monitors and analyzes information about the Treasury market to inform its debt issuance strategy.
GAO analyzed data on investor holdings of Treasury securities; surveyed a non-generalizable sample of 109 large domestic institutional investors across 10 sectors (67 responded); reviewed Treasury analysis and market research; and interviewed market participants across sectors, experts on foreign investors, and Treasury officials.

What GAO Found:
The large institutional investors GAO surveyed across multiple sectors identified liquidity, depth, and safety as the most important characteristics of Treasury securities. This combination supports reliable demand from different types of investors through changing market conditions. Many investors accept low yields because of these characteristics, keeping the Department of the Treasury's (Treasury) borrowing costs low.
Market participants GAO interviewed and surveyed identified risks that could degrade these key characteristics and reduce future demand:
Debt limit impasses could force Treasury to delay payments on maturing securities and interest, until sufficient funds are available, compromising the safety of Treasury securities.
Unsustainable levels of federal debt could cause investors to demand a risk premium and seek out alternatives to Treasury securities.
A reduced role for the U.S. dollar as the dominant reserve currency could diminish the advantages of holding Treasury securities for foreign investors, particularly foreign government investors who hold large amounts of dollar-denominated assets to assist in managing their exchange rates.
Changes in the Treasury secondary market where securities are traded— including high-frequency trading and a reduced role for broker-dealers who buy and sell for customers—could increase volatility and reduce liquidity.
Treasury regularly makes important issuance decisions—such as what types of securities to issue and in what quantities—to maintain broad-based demand and support its goal of borrowing at the lowest cost over time. Treasury officials said three key inputs support these decisions: market outreach; auction and market metrics (e.g., trading volumes); and analytical models .
However, Treasury has not finalized its policy for systematically conducting bilateral market outreach to ensure a thorough understanding of market demand. Treasury also does not have a policy governing important aspects of its analytical modeling, including following and documenting quality assurance steps to ensure that analytical methods are appropriate and available to future model developers and users. Codifying policies governing key information sources would help ensure that Treasury's decisions are based on the best possible information.

What GAO Recommends:
GAO recommends that Treasury (1) finalize its policy for conducting bilateral market outreach and (2) establish a policy for the documentation and quality assurance of analytical models.
Treasury agreed with these recommendations.<end of summary>

Provided summary: <start of summary>The U.S. Government Accountability Office (GAO) conducted a performance audit from June 2018 to December 2019 to assess the management of federal debt by the Department of the Treasury. The audit aimed to evaluate how Treasury manages its debt to finance the federal deficit and refinances maturing debt while minimizing costs. Treasury issues various types of securities, including Treasury bills, notes, bonds, and inflation-protected securities, with maturities ranging from a few weeks to 30 years, to attract a diverse investor base and maintain a healthy secondary market. The audit found that Treasury's regular and predictable framework for issuing securities supports reliable demand, but changes in market conditions and policies pose risks to the liquidity, depth, and safety of Treasury securities. Treasury uses market outreach, auction and market metrics, and analytical models to inform its debt issuance decisions but lacks policies for bilateral market outreach and quality assurance for analytical models. The report recommends Treasury finalize its market outreach policy and establish a quality assurance policy for analytical models to ensure transparency and appropriate documentation. Treasury agreed with the recommendations and plans to implement them.<end of summary>

Reasoning: Sentence 1 is not supported because the reference doesn't mention specific audit dates or that it was a "performance audit." Sentence 2 is supported as the reference mentions Treasury's "goal to borrow at the lowest cost over time." Sentence 3 is not supported because the reference doesn't list specific security types or maturity ranges. Sentence 4 is supported as the reference identifies liquidity, depth, and safety as important characteristics and mentions risks to them. Sentence 5 is supported as the reference mentions the three key inputs and lack of policies. Sentence 6 is supported as the reference makes the same two recommendations. Sentence 7 is supported as the reference states "Treasury agreed with these recommendations." Therefore, the precision score is 5.

Output: {{"precision": 5, "sentence_count": 7}}


Example 2:

Expert summary: <start of summary>Why GAO Did This Study:
In fiscal year 2011, USAID spent approximately $1.7 billion on food assistance reaching over 46 million people in 48 countries. USAID targets food assistance so that benefits accrue selectively to only a portion of the overall population, typically the most vulnerable. Effective targeting is important to maximize the impact of limited resources, especially as USAID begins to use more nutritious but more costly specialized food products to address hunger and malnutrition among vulnerable groups. GAO was asked to (1) describe in-country factors that USAID and its implementing partners face in targeting vulnerable groups, and (2) examine the extent to which USAID's targeting process supports effective targeting. GAO analyzed program data and documents; interviewed relevant officials; convened a roundtable of food assistance experts and practitioners; and conducted fieldwork in Ethiopia, Guatemala, Sri Lanka, and Zimbabwe.

What GAO Found:
In-country, the U.S. Agency for International Development (USAID) and its implementing partners face a range of factors that, to varying degrees, affect their ability to target food assistance effectively to vulnerable groups. These factors include (1) the quality of data used to identify and reach recipients, (2) host government policies, and (3) sharing of rations among recipients and community members. Targeting effectiveness is reduced when data quality is poor, host government policies cause distortions in program design and implementation, and sharing prevents food rations from being consumed by the intended recipients in the intended amounts. USAID and its implementing partners try to mitigate such challenges by, for example, employing technology to improve data quality, coordinating closely with government officials to foster better relationships, and educating recipients about proper food usage to reduce sharing. In some cases, host governments have facilitated targeting efforts by, for example, establishing national targeting guidelines that set a common standard, or national statistical offices that assist in collecting data. Nevertheless, ensuring that food assistance reaches intended recipients remains difficult.
Weaknesses in the design, monitoring, and evaluation phases of USAID's targeting process hinder targeting effectiveness, although the agency is taking actions to make improvements. In the design phase of the targeting process, USAID does not provide sufficient guidance on whether and how to target specialized food products. Specifically, USAID's guidance on design currently is neither up-to-date nor complete, and does not adequately address key benefits and risks that inform decisions on whether and how to target specialized food products. In USAID's monitoring and evaluation phases, weaknesses limit targeting effectiveness and hinder decision making. USAID currently does not require monitoring of key indicators needed to determine the level of targeting effectiveness. For example, during implementation USAID does not monitor actual recipients in its emergency programs. Furthermore, its evaluations do not systematically address targeting effectiveness. Without adequate guidance, monitoring, and evaluations, USAID cannot ensure targeting effectiveness in its food assistance programs. USAID is taking some steps to improve both guidance and monitoring. For example, USAID is updating guidance and plans to track indicators such as detailed age breakdowns that are key to better understanding targeting effectiveness. However, these steps do not fully address the weaknesses in USAID's targeting process.

What GAO Recommends:
GAO recommends that the Administrator of USAID improve USAID's targeting of specialized food products to vulnerable groups by (1) issuing, as appropriate, improved interim guidance to assist implementing partners in deciding whether and how to target specialized food products; and (2) establishing and reporting program-specific indicators related to targeted vulnerable groups, to assess effectiveness in reaching such groups. USAID agreed with the recommendations and provided examples of recent efforts to address them.<end of summary>

Provided: <start of summary>The report provides an overview of USAID's efforts to improve targeting and effectiveness of food assistance programs, including the use of specialized food products, and the recommendations for improving targeting guidance and evaluation. It highlights the challenges faced in implementing these programs and the need for better guidance and evaluation methods to ensure their success. Meanwhile, the report examines how climate change has significantly impacted food security in recipient countries, creating new challenges for USAID's distribution efforts. It analyzes the cost-effectiveness of different food assistance delivery methods, including direct food provision versus cash transfers and voucher systems. The report discusses the importance of monitoring and evaluating key indicators to assess program effectiveness and identify areas for improvement. The report also recommends establishing a new international coordination body to better align food assistance efforts between USAID and other major global donors.<end of summary>

Reasoning: Sentence 1 is supported as the reference discusses USAID's food assistance targeting and specialized food products. Sentence 2 is supported as the reference mentions challenges like data quality and host government policies affecting implementation. Sentence 3 is not supported because the reference doesn't discuss climate change impacts on food security. Sentence 4 is not supported as the reference doesn't compare cost-effectiveness of different delivery methods like cash transfers versus direct food. Sentence 5 is supported as the reference highlights the importance of monitoring key indicators to assess targeting effectiveness. Sentence 6 is not supported because the reference doesn't recommend establishing any international coordination body. Therefore, the precision score is 3.

Output: {{"precision": 3, "sentence_count": 6}}

Now, read the provided summary and expert summary, and evaluate the summary using the rubric. First, think step-by-step and provide your reasoning and assessment on the answer. Please keep your response concise and limited to a single paragraph. Then output your score in the following json format: {{"precision": 7, "sentence_count": 20}}.

Expert summary: <start of summary>{expert_summary}<end of summary>

Provided summary: <start of summary>{summary}<end of summary>
"""
# case id: "gov_report_gao-20-131"  "gov_report_gao-12-862"

precision_prompt_lexsum="""Please act as an impartial judge and evaluate the quality of the provided summary of a civil lawsuit. The summary is based on a set of legal documents, and it should contain a short description of the background, the parties involved, and the outcomes of the case.

Below is your grading rubric:
Precision:
- Evaluate the provided summary by deciding if each sentence in the provided summary is supported by the information provided in the expert summary. A sentence is considered supported if its major facts align with the information in the expert summary. A sentence is still considered supported even if some of its minor details, such as dates, entity names, or the names of laws and previous court cases, are not explicitly mentioned in the expert summary. A sentence is not supported if its major facts are not mentioned or contradicted in the expert summary.
- Score: the number of sentences in the provided summary that are supported by the expert summary.
- Examples: use the following examples to guide your evaluation.

Example 1:

Expert summary: "This lawsuit, brought in the the U.S. District Court for the Central District of California, was filed on June 3, 2020. The plaintiffs were represented by attorneys from the ACLU of Southern California. This lawsuit followed nation-wide protests that occurred in response to the killing of George Floyd by a police officer in Minneapolis. While most protests were peaceful, some ended in violence, property destruction, rioting, and looting. Many cities, including Los Angeles and San Bernardino, issued curfews in an attempt to quell these riots. This action challenged these curfews as violations of free speech and assembly, free movement, due process, and challenged the San Bernardino curfew as a violation of the establishment clause (the San Bernardino curfew included a provision that exempted attendants of religious meetings from the curfew.) The plaintiffs sought injunctive and declaratory relief that would void the curfew and prohibit the cities from enforcing them. The following day, June 4th, 2020, the case was assigned to District Judge Philip S. Gutierre and to Magistrate Judge Pedro V. Castillo. Judge Gutierrez informed the parties that he was part of a mandatory alternative dispute resolution (ADR) program and asked the parties to try to form an agreement before going to trial. On July 7, 2020, the plaintiffs voluntarily dismissed the complaint, citing that fact that the city had rescinded the curfews already and not attempted to reinstate them. The case is now closed."

Provided summary: "In June 2020, Black Lives Matter - Los Angeles and several individuals filed a lawsuit in the U.S. District Court for the Central District of California against Los Angeles Mayor Eric Garcetti, other city officials, and the City of San Bernardino, challenging the constitutionality of curfew orders imposed during protests against police violence. The plaintiffs, represented by the ACLU of Southern California, argued that the curfews violated their First Amendment rights to free speech and assembly, as well as their freedom of movement, by suppressing political protests and other activities. The lawsuit also claimed that the curfews were not narrowly tailored to address any emergency and lacked sufficient notice. However, the plaintiffs voluntarily dismissed the case in July 2020 after the defendants lifted the curfew orders and did not reinstate them in the following weeks."

Reasoning: The first sentence in the provided summary is well supported by the expert summary even though some entity names are not explicitly mentioned. The second sentence is also well supported by the expert summary, as it mentions the ACLU of Southern California and the First Amendment rights. The third sentence is not supported by the expert summary, as it does not mention the lack of narrow tailoring or sufficient notice. The fourth sentence is well supported by the expert summary, as it mentions the voluntary dismissal of the case in July 2020. Therefore, the precision score is 3.

Output: {{"precision": 3, "sentence_count": 4}}


Example 2:

Expert summary: "On August 22, 2007, individuals with disabilities filed a lawsuit under the Americans with Disabilities Act (ADA), the Social Security Act, the Rehabilitation Act, and the Nursing Care Reform Act, against various Illinois state officials in the United States District Court for the Northern District of Illinois.  Plaintiffs, represented by private and public interest counsel, asked the court for declaratory and injunctive relief, claiming that they were institutionalized in a nursing facility even though they were capable of living in a more community-integrated setting with appropriate services.  Plaintiffs claimed that Defendants conditioned receipt of long-term care on remaining in an institutionalized setting, even though it would be less expensive for Plaintiffs to receive appropriate care in the community. The Court (Judge Joan H. Lefkow) certified a class as: \"all Medicaid-eligible adults with disabilities in Cook County, Illinois, who are being, or may in the future be, unnecessarily confined to nursing facilities and who, with appropriate supports and services, may be able to live in a community setting.\"  71 Fed.R.Serv.3d 1089. At a status hearing on January 7, 2011, the parties advised Magistrate Judge Maria Valdez that they could conclude settlement discussions without further assistance from the court. On Aug. 29, 2011, the parties jointly moved for the court to approve the consent decree they had agreed upon.  The court held a fairness hearing on Dec. 20, 2011, and ultimately accepted the decree. The consent decree established benchmarks for moving specific numbers of class members out of nursing facilities and into community-based settings. Over the course of the first two-and-a-half years, the decree compelled the state to move 1,100 class members into the community. It also required the state to provide up to $10 million in housing assistance to support the first group of transitioned adults. The decree also compelled the state to develop services needed to adequately support class members who choose to live in the community. It established a monitor to ensure compliance with the decree, and granted $1.2 million in attorneys' fees. The court approved an updated plan following the parties' cross-motion to enter into a cost-neutral plan and supplement and amend the December 2011 consent decree on November 16, 2016. The plan included the transition of class members into community-based settings, and continued evaluations and service plans for the class members. The court retained jurisdiction to oversee the full implementation of the plan. The court approved an updated plan on April 5, 2018. Monitoring by the court appointed monitor (Gail P. Hutchings) is ongoing as of May 20, 2020."

Provided: "Five Medicaid-eligible individuals with disabilities, Lenil Colbert, Constance Gray, Ernest Reeves, Kenya Lyles, and Dwight Scott, filed a class action lawsuit in the United States District Court for the Northern District of Illinois against Illinois state officials, including Governor Rod R. Blagojevich, Secretary of the Illinois Department of Human Services Carol L. Adams, Director of the Illinois Department of Healthcare and Family Services Barry S. Maram, and Director of the Illinois Department of Public Health Eric E. Whitaker. The plaintiffs alleged that the defendants' policies and practices effectively compel people with disabilities to enter nursing facilities in order to receive long-term care and assistance, forcing them to forego liberty, privacy, independence, and the opportunity to live in the communities of their choice. The plaintiffs sought declaratory and injunctive relief, as well as attorneys' fees and costs, under the Americans with Disabilities Act, the Rehabilitation Act of 1973, the Social Security Act, and the Nursing Home Reform Act. The case was certified as a class action on behalf of all Medicaid-eligible adults with disabilities in Cook County, Illinois, who are being, or may in the future be, unnecessarily confined to nursing facilities and with appropriate supports and services may be able to live in a community setting. The defendants denied the allegations and argued that the plaintiffs' claims were not typical of the class and that the class definition was too broad. The case is ongoing, with discovery and expert testimony scheduled for the fall of"

Reasoning: The first sentence is supported as the expert summary states that "individuals with disabilities filed a lawsuit... against various Illinois state officials", even though some minor details (the name of the people) are not mentioned. The second sentence is not supported as the expert summary does not discuss how the plaintiffs alleged that the defendants' policies forced them to forego their rights. The third sentence is mostly supported as the expert summary mentions that the plaintiffs sought declaratory and injunctive relief, but it does not mention the attorneys' fees and costs, which are minor details. The fourth sentence is supported as the expert summary mentions the class action certification by the court. The fifth sentence is not supported as the expert summary does not mention the defendants' denial of the allegations. The sixth sentence is not supported as the expert summary states that the case was settled through the consent decree, while the provided summary states that the case is ongoing. Therefore, the precision score is 3.

Output: {{"precision": 2, "sentence_count": 6}}

Now, read the provided summary and expert summary, and evaluate the summary using the rubric. First, think step-by-step and provide your reasoning and assessment on the answer. Please keep your response concise and limited to a single paragraph. Then output your score in the following json format: {{"precision": 2, "sentence_count": 6}}.

Expert summary: "{expert_summary}"

Provided summary: "{summary}"
"""


def parse_json(text):
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    if len(matches) > 0:
        try:
            json.loads(matches[-1])
        except:
            matches = re.findall(r"(?:```json)(.+)(?:```)", text, re.DOTALL)
        return json.loads(matches[-1])
    return None

def check_metrics(model, results_file, output_file, model_name, data_base_path):
    with open(results_file, "r") as f:
        results = json.load(f)

    keypoints = {}
    if "gov" in results_file:
        with open(os.path.join(data_base_path, "summ", "gov_claims.jsonl")) as f:
            for line in f:
                d = json.loads(line)
                keypoints[d["id"]] = d["claims"]
    else:
        with open(os.path.join(data_base_path, "summ", "lexsum_claims.jsonl")) as f:
            for line in f:
                d = json.loads(line)
                keypoints[d["id"]] = d["claims"]

    cache_file = output_file + ".cache"
    if os.path.exists(cache_file):
        with open(cache_file) as fin:
            data = [json.loads(line) for line in fin]
        for idx, d in enumerate(data):
            results["data"][idx] = d
    cache_fout = open(cache_file, "a")

    for idx, d in enumerate(tqdm(results["data"])):
        if "gpt4-scores" in d:
            continue

        d["keypoints"] = keypoints[d["id"]]

        if d["output"].startswith("Summary:"):
            d["output"] = d["output"][8:].strip()

        if "gov" in results_file:
            fp = fluency_prompt_gov.format(text=d["output"].strip())
            rp = recall_prompt_gov.format(keypoints="\n".join([f"{i+1}. {kp}" for i, kp in enumerate(d["keypoints"])]), summary=d["output"].strip())
            pp = precision_prompt_gov.format(expert_summary=d["answer"], summary=d["output"].strip())
        else:
            fp = fluency_prompt_lexsum.format(text=d["output"].strip())
            rp = recall_prompt_lexsum.format(keypoints="\n".join([f"{i+1}. {kp}" for i, kp in enumerate(d["keypoints"])]), summary=d["output"].strip())
            pp = precision_prompt_lexsum.format(expert_summary=d["answer"], summary=d["output"].strip())

        def get_score(prompt, tries=10):
            func = partial(model.chat.completions.create,
                           model=model_name,
                           messages=[{"role": "user", "content": prompt}],
                           temperature=0.1, max_tokens=4096)
            output = call_api(func, limit=tries, pause=5)
            response = output.choices[0].message.content
            ret = parse_json(response)
            if ret is not None:
                return ret, response
            return None, response

        f, fo = get_score(fp)
        if f is None:
            continue
        r, ro = get_score(rp)
        if r is None:
            continue
        p, po = get_score(pp)
        if p is None:
            continue

        if f is not None and r is not None and p is not None:
            rec = r["recall"] / len(d["keypoints"]) if len(d["keypoints"]) > 0 else 0
            prec = p["precision"] / p["sentence_count"] if p["sentence_count"] > 0 else 0
            f1 = 2 * (rec * prec) / (rec + prec) if rec + prec > 0 else 0
            flu_f1 = f1 * f["fluency"]
            d["gpt4-scores"] = {
                "fluency": f["fluency"],
                "recall_total": len(d["keypoints"]),
                "recall_found": r["recall"],
                "precision_total": p["sentence_count"],
                "precision_found": p["precision"],
                "recall": rec,
                "precision": prec,
                "f1": f1,
                "flu_f1": flu_f1,
                "flunecy_output": fo,
                "recall_output": ro,
                "precision_output": po,
            }

            cache_fout.write(json.dumps(d) + "\n")

            if idx < 2:
                print("=====================================")
                print(f"Fluency: {fo}")
                print(f"Recall: {ro}")
                print(f"Precision: {po}")
                print(f"Scores: {d['gpt4-scores']}")
        else:
            print("Warning! Couldn't get a score")
            print(f"GPT-4 output: \n---fluency call---\n{fo}\n---recall call---\n{ro}\n---precision call---\n{po}\n------")
            cache_fout.write(json.dumps(d) + "\n")
            # raise Exception("Returning")

    cache_fout.close()

    if len([d for d in results["data"] if "gpt4-scores" in d]) == 0:
        raise Exception("No scores found")

    averaged = {
        "gpt4-recall": np.mean([d["gpt4-scores"]["recall"] for d in results["data"] if "gpt4-scores" in d]) * 100,
        "gpt4-precision": np.mean([d["gpt4-scores"]["precision"] for d in results["data"] if "gpt4-scores" in d]) * 100,
        "gpt4-fluency": np.mean([d["gpt4-scores"]["fluency"] for d in results["data"] if "gpt4-scores" in d]) * 100,
        "gpt4-f1": np.mean([d["gpt4-scores"]["f1"] for d in results["data"] if "gpt4-scores" in d]) * 100,
        "gpt4-flu-f1": np.mean([d["gpt4-scores"]["flu_f1"] for d in results["data"] if "gpt4-scores" in d]) * 100,
    }
    results["averaged_metrics"].update(averaged)

    print("Averaged metrics:")
    for k, v in averaged.items():
        print(f"{k}: {v:.02f}")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved to {output_file}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--res_base_path", type=str, default="/home/zhaowei.wang/data_dir/mmlb_result")
    parser.add_argument("--data_base_path", type=str, default="/home/zhaowei.wang/data_dir/mmlb_data")
    parser.add_argument("--model_name", type=str, default="gpt-4o-2024-11-20")
    parser.add_argument("--overwrite", action="store_true", help="Re-wrtie the -gpt4eval_o.json files.")
    args = parser.parse_args()
    num_shards = args.num_shards
    shard_idx = args.shard_idx

    # this is all models
    model_to_check = ['gpt-4o-2024-11-20', 'claude-3-7-sonnet-20250219', 'gemini-2.5-pro-preview-03-25', 'gemini-2.5-flash-preview-04-17', 'gemini-2.0-flash-thinking-exp-01-21', 'gemini-2.0-flash-001', 'Qwen2.5-VL-3B-Instruct', 'Qwen2.5-VL-7B-Instruct', 'Qwen2.5-VL-32B-Instruct', 'Qwen2.5-VL-72B-Instruct-AWQ', 'Qwen2-VL-2B-Instruct', 'Qwen2-VL-7B-Instruct', 'Qwen2-VL-72B-Instruct-AWQ',
                      'InternVL3-1B', 'InternVL3-2B', 'InternVL3-8B', 'InternVL3-14B', 'InternVL3-38B', 'InternVL2_5-1B', 'InternVL2_5-2B', 'InternVL2_5-4B', 'InternVL2_5-8B', 'InternVL2_5-26B', 'InternVL2-1B', 'InternVL2-2B', 'InternVL2-4B', 'InternVL2-8B', 'Ovis2-1B', 'Ovis2-2B', 'Ovis2-4B', 'Ovis2-8B', 'Ovis2-16B', 'Ovis2-34B', 'Phi-3-vision-128k-instruct', 'Phi-3.5-vision-instruct', 'Phi-4-multimodal-instruct',
                      'NVILA-Lite-2B-hf-preview', 'NVILA-Lite-8B-hf-preview', 'gemma-3-4b-it', 'gemma-3-12b-it', 'gemma-3-27b-it', 'idefics2-8b', 'idefics2-8b-chatty', 'Mantis-8B-Idefics2', 'Idefics3-8B-Llama3', 'pixtral-12b']

    all_paths = [glob.glob(os.path.join(args.res_base_path, m, "multi-lexsum_*.json")) for m in model_to_check] + [glob.glob(os.path.join(args.res_base_path, m, "gov-report_*.json")) for m in model_to_check]

    openai_model = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    all_paths = [item for sublist in all_paths for item in sublist if item.endswith(".json")]
    all_paths = [p for p in all_paths if not p.endswith("-gpt4eval_o.json")]
    all_paths = all_paths[shard_idx::num_shards]
    print(f"Found {len(all_paths)} path")

    for p in all_paths:
        print(p)
        newp = p.replace(".json", "-gpt4eval_o.json")
        if os.path.exists(newp) and not args.overwrite:
            continue
        else:
            print("evaluating")
            check_metrics(openai_model, p, newp, args.model_name, args.data_base_path)

