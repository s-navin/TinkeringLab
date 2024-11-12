# NLP-based Query-Response Tool

## Motivation 
Business stakeholders need a tool to simplify the FAQ-based Query-Response interactions of customers and 
improve the customer experience on their website. In particular, rather than customers having to 
navigate through FAQ pages, they would prefer customers to be able to type in a question and get a relevant answer.

e.g. Chiru just recently bought an EV and wants to use Shell’s Recharge Network to charge his EV during a road trip.    
- Customer query : I want to download the Shell Recharge app, where can I find it?    
- System’s answer: From Google Play and App store    

## Business Requirements 
The stakeholders are looking for a Proof of Concept for an NLP-based Query-Response tool for the purpose specified above. They want to be able to
type in a question and test if the model is able to return a relevant answer. They have provided a folder (.\faq) with all the FAQs scraped from their website in html format.

#### Specifications:   
- Implement the tool using:   
      o Tech-Stack : Python   
      o Framework  : any NLP/ML libraries    
      o Deliverable: a PoC that runs in a terminal (CLI).    
        - Optional: Create a very simple browser-GUI where users (customers) can:   
            o Type in a query   
            o Get a relevant answer   
            o Recommendation: build a streamlit-app that can be launched locally   
