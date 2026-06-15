def main() -> None:
    '''
    Main function to run the RAG application.
    '''

    try:
        from fire import Fire
        from .rag import RAG

        Fire(RAG)
    except BaseException:
        print('\n[\033[95mRAG\033[0m] \033[3mExiting...\033[0m')


if __name__ == '__main__':
    main()
