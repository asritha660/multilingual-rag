from pypdf import PdfReader

# Open the PDF file
reader = PdfReader("sample.pdf")

# Tell us how many pages it has
print("Number of pages:", len(reader.pages))

# Grab the text from the very first page and print it
first_page = reader.pages[0]
text = first_page.extract_text()
print("\n--- TEXT FROM PAGE 1 ---\n")
print(text)