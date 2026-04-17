# FOI Search Tool
A natural language tool allowing citizens to ask questions about the council based on previously answered FOIs.

# Requirements
Poetry to create env
.env.example

# Workflow
1. Text is cleaned to remove ASCII characters, unneeded text like address and so on 
2. Ingested (chunked, embeded and stored)
3. Retreived (clustering and prompt)
4. Evaluation and testsSS

# Data 
Sourced from https://opendata.camden.gov.uk/Your-Council/Camden-Freedom-Of-Information-Responses-Search/fkj6-gqb4/data_preview
Using the Document Text column, instead of ingesting pdfs ect

