import torch

def get_docs(embedding: torch.Tensor):
    # load document_embedding_index
    # lookup query embedding in document_embedding_index

    # return top/bottom docs as a list of strings
    rel_docs = [
        "doc1",
        "doc2",
        "The shift to remote work has fundamentally changed how businesses operate, paving the way for a more flexible, dynamic workforce. With advances in technology and the pandemic accelerating its adoption, remote work has allowed organizations to tap into a global talent pool, transcending geographical limitations. This transformation offers significant benefits: employees can achieve better work-life balance, and companies can reduce overhead costs associated with physical office spaces. However, this shift has also introduced challenges, particularly around team cohesion, communication, and mental health. As teams become more dispersed, maintaining a strong company culture requires intentional efforts from leadership. Companies are increasingly relying on virtual collaboration tools like Slack, Zoom, and project management platforms to facilitate communication and enhance productivity. Mental health support has also become a priority, with many organizations offering wellness programs and resources. While remote work is not without its complexities, it is clear that the benefits are reshaping the future of work. As businesses adapt, they will likely adopt hybrid models, allowing employees to split their time between the office and home. This evolution in work culture represents an exciting opportunity to create a more inclusive and sustainable approach to employment worldwide.",
        "doc4",
        "doc5",
    ]

    rel_docs_sim = [0.8, 0.7, 0.6, 0.5, 0.4]

    irel_docs = ["doc6", "doc7", "doc8", "doc9", "doc10"]

    irel_docs_sim = [0.3, 0.2, 0.1, 0.09, 0.08]

    return rel_docs, rel_docs_sim, irel_docs, irel_docs_sim
