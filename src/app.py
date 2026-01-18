from rag_chain import build_rag_chain

def main():
    rag = build_rag_chain()

    while True:
        query = input("\nAsk a cancer-related question (or exit): ").strip()
        if query.lower() == "exit":
            break

        response = rag.invoke(query)

        print("\nAnswer:")
        print(response)

if __name__ == "__main__":
    main()
