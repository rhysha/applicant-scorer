from pyresparser import ResumeParser
data = ResumeParser('/home/rhysha/applicant-scoring/resume.pdf').get_extracted_data()
print(data)
