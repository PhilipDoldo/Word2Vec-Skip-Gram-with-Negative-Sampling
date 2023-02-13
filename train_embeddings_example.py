import lib.module as module
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(0)


if __name__ == "__main__":
    
    data = "Many cities contain tall buildings such as skyscrapers or offices, which can also be skyscrapers. Skyscrapers are an example of tall buildings, but other kinds of tall buildings exist, such as banks. Banks are not actually typical examples of tall buildings as far as I am aware, but tall buildings can come in all kinds of shapes and sizes, except for sizes which are not very high in which case we would no longer classify the corresponding not tall buildings as tall buildings. Many people live in tall buildings and a lot of the time you will see birds fly onto tall buildings. There are many sights you can see from tall buildings, provided that you are high enough in the tall buildings. I know that I used the plural form of tall buildings although realistically you will probably only be within a single tall buildings at a time (I did it again) but I want tall buildings to show up in this text a lot."
    
    # process
    words = list(module.process_text(data).split())
    
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
    
    
    learning_schedule = [.02*(.99)**i for i in range(100)]
    losses = []
    
    losses.append( module.get_loss(training_data, center_embeddings, context_embeddings) )
    
    for epoch, learning_rate in enumerate(learning_schedule):
        
        print(f"==== EPOCH {epoch+1} ====")
        
        # train embeddings
        center_embeddings, context_embeddings, loss = module.train_embeddings(
            training_data=training_data,
            center_embeddings=center_embeddings,
            context_embeddings=context_embeddings,
            batch_size=len(training_data)//35,
            learning_rate=learning_rate,
            num_epochs=1,
            verbose=True
            )
        
        losses += loss

    
    plt.plot( range(0,len(learning_schedule)+1), losses )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    
    plt.plot( range(len(learning_schedule)+1), losses )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(top=losses[min(3, len(losses)-1)])
    plt.show()
    
    final_embeddings = center_embeddings + context_embeddings
    
    w='tall'
    print(f"Closest words to {w}:\n{module.find_closest_words(w, final_embeddings).iloc[0:10]}")
    
    w='buildings'
    print(f"Closest words to {w}:\n{module.find_closest_words(w, final_embeddings).iloc[0:10]}")
    
    module.plot_2d_grid(final_embeddings)
    
    """
    From the 2d grid plot, we can see that the word embeddings have formed some
    sort of structure and words such as "tall" and "buildings" are close to
    each other which lines up with the fact that the phrase "tall buildings"
    was intentionally placed within the training text multiple times. This 
    similarity between words being captured was achieved without even trying to
    optimize training very much.
    """