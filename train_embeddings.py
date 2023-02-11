import lib.module as module


if __name__ == "__main__":
    
    # TODO: load data as a big string
    # data =
    
    data = "I walked to the store to buy some food which I stored in my house in the cabinet. I often try to buy things at the store so that I never run out of supplies. Buying items from the store requires money which I can use to pay with in exchange for goods. The only annoying thing about going to the store is that I need to get into my car and drive there. I'm not a big fan of driving to the store, but I try to be a responsible adult and perform this duty on a somewhat frequent basis. My ability to consistently motivate myself to go to the store has played a huge role in my success as a business owner. One might initially think that going to the store is simply a trivial task that would not eventually lead to me becoming an incredibly successful business owner, but after further reflection this individual might come to the conclusion that going to the store is actually a very important part of becoming a millionaire. It teaches you all of the skills you need to get rich, such as perseverance, moving quickly, and consistency. Do not sleep on going to the store, it will lead to your success."
    
    # process
    words = module.process_text(data).split()
    
    # initialize word embeddings
    center_embeddings, context_embeddings = module.initialize_embeddings(
        words=words,
        embedding_dimension=2,
        mean=0,
        std=1
        )
    
    # initialize training data
    training_data = module.initialize_training_data(
        words=words,
        window_size=3,
        k=2,
        scale=0.75
        )
    
    center_embeddings, context_embeddings = module.train_embeddings(
        training_data=training_data,
        center_embeddings=center_embeddings,
        context_embeddings=context_embeddings,
        batch_size=1,
        learning_rate=.01,
        num_epochs=3,
        verbose=True
        )
