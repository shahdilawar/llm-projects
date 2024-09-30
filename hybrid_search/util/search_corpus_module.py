'''
This file serves as a module for data prep for AI semantic search.
'''
class SearchCorpusModule:
      #
      # From "Through the looking Glass" by Lewis Caroll
      #
      jabberwocky = """
      ’Twas brillig, and the slithy toves
            Did gyre and gimble in the wabe:
      All mimsy were the borogoves,
            And the mome raths outgrabe.

      “Beware the Jabberwock, my son!
            The jaws that bite, the claws that catch!
      Beware the Jubjub bird, and shun
            The frumious Bandersnatch!”

      He took his vorpal sword in hand;
            Long time the manxome foe he sought—
      So rested he by the Tumtum tree
            And stood awhile in thought.

      And, as in uffish thought he stood,
            The Jabberwock, with eyes of flame,
      Came whiffling through the tulgey wood,
            And burbled as it came!

      One, two! One, two! And through and through
            The vorpal blade went snicker-snack!
      He left it dead, and with its head
            He went galumphing back.

      “And hast thou slain the Jabberwock?
            Come to my arms, my beamish boy!
      O frabjous day! Callooh! Callay!”
            He chortled in his joy.

      ’Twas brillig, and the slithy toves
            Did gyre and gimble in the wabe:
      All mimsy were the borogoves,
            And the mome raths outgrabe.

      """

      #
      # The beginning of the "Tale of two cities", by Charles Dickens
      #
      best_of_times = """
      It was the best of times, it was the worst of times, 
      it was the age of wisdom, it was the age of foolishness, 
      it was the epoch of belief, it was the epoch of incredulity, 
      it was the season of light, it was the season of darkness, 
      it was the spring of hope, it was the winter of despair, 
      we had everything before us, we had nothing before us, 
      we were all going direct to heaven, 
      we were all going direct the other way–in short, 
      the period was so far like the present period, 
      that some of its noisiest authorities insisted on its being received, 
      for good or for evil, in the superlative degree of comparison only.
      """

      #
      # From the "Tale of two cities" by Charles Dickens
      #
      mystery = """
      A wonderful fact to reflect upon, that every human creature is 
      constituted to be that profound secret and mystery to every other. 
      """

      #
      # A poignant passage from the "Tale of two cities", by Charles Dickens
      #
      last_dream = """
      I wish you to know that you have been the last dream of my soul. 
      In my degradation I have not been so degraded but that the sight 
      of you with your father, and of this home made such a home by you, 
      has stirred old shadows that I thought had died out of me. 
      Since I knew you, I have been troubled by a remorse that I 
      thought would never reproach me again, and have heard whispers 
      from old voices impelling me upward, that I thought were silent 
      for ever. I have had unformed ideas of striving afresh, beginning anew, 
      shaking off sloth and sensuality, and fighting out the abandoned fight. 
      A dream, all a dream, that ends in nothing, and leaves the sleeper 
      where he lay down, but I wish you to know that you inspired it.
      """

      mark_twain_dog = """
      The dog is a gentleman; I hope to go to his heaven not man's.
      """

      einstein = """If a man aspires towards a righteous life, his first act of abstinence is from injury to animals."""

      tweedledee  = """
      Tweedledum and Tweedledee: She then meets the fat twin brothers 
      Tweedledum and Tweedledee, whom she knows from the nursery rhyme. 
      After reciting the long poem "The Walrus and the Carpenter", 
      they draw Alice's attention to the Red King—loudly snoring away 
      under a nearby tree—and maliciously provoke her with idle philosophical 
      banter that she exists only as an imaginary figure in the Red King's dreams. 
      Finally, the brothers begin suiting up for battle, only to be frightened 
      away by an enormous crow, as the nursery rhyme about them predicts.
      """

      goldens_1 = """
      Golden retrievers are not bred to be guard dogs, and considering the size of their hearts and their irrepressible joy and life, they are less likely to bite than to bark, less likely to bark than to lick a hand in greeting. In spite of their size, they think they are lap dogs, and in spite of being dogs, they think they’re also human, and nearly every human they meet is judged to have the potential to be a boon companion who might at any moment, cry, “Let’s go!” and lead them on a great adventure.
      """

      goldens_2 = """
      If you’re lucky, a golden retriever will come into your life, steal your heart, and change everything
      """

      goldens_3 = """
      My friend Phil has a theory that the Lord, having made teenagers, felt constrained to make amends and so created the golden retriever.
      """

      dog_soul = """
      If you don’t believe that dogs have souls, you haven’t looked into their eyes long enough.
      """

      keats = """
      A thing of beauty is a joy for ever:
      Its loveliness increases; it will never
      Pass into nothingness; but still will keep
      A bower quiet for us, and a sleep
      Full of sweet dreams, and health, and quiet breathing.
      Therefore, on every morrow, are we wreathing
      A flowery band to bind us to the earth,
      Spite of despondence, of the inhuman dearth
      Of noble natures, of the gloomy days,
      Of all the unhealthy and o'er-darkn'd ways
      Made for our searching: yes, in spite of all,
      Some shape of beauty moves away the pall
      From our dark spirits. Such the sun, the moon,
      Trees old and young, sprouting a shady boon
      For simple sheep; and such are daffodils
      With the green world they live in; and clear rills
      That for themselves a cooling covert make
      'Gainst the hot season; the mid-forest brake,
      Rich with a sprinkling of fair musk-rose blooms:
      And such too is the grandeur of the dooms
      We have imagined for the mighty dead;
      An endless fountain of immortal drink,
      Pouring unto us from the heaven's brink
      """

      attention = """
      The dominant sequence transduction models are based on 
      complex recurrent or convolutional neural networks in an encoder-decoder configuration. 
      The best performing models also connect the encoder and decoder through 
      an attention mechanism. We propose a new simple network architecture, 
      the Transformer, based solely on attention mechanisms, 
      dispensing with recurrence and convolutions entirely. 
      Experiments on two machine translation tasks show these models 
      to be superior in quality while being more parallelizable 
      and requiring significantly less time to train. 
      Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, 
      improving over the existing best results, including ensembles by over 2 BLEU. 
      On the WMT 2014 English-to-French translation task, our model establishes 
      a new single-model state-of-the-art BLEU score of 41.8 after training for 
      3.5 days on eight GPUs, a small fraction of the training costs of the 
      best models from the literature. We show that the Transformer 
      generalizes well to other tasks by applying it successfully to 
      English constituency parsing both with large and limited training data.

      """

      backprop = """
      In machine learning, backpropagation
      (backprop,[1] BP) is a widely used
      algorithm for training feedforward
      artificial neural networks.
      Generalizations of backpropagation
      exist for other artificial neural
      networks (ANNs), and for functions
      generally. These classes of algorithms
      are all referred to generically as
      "backpropagation".[2] In fitting a
      neural network, backpropagation
      computes the gradient of the loss
      function with respect to the weights of
      the network for a single input–output
      example, and does so efficiently,
      unlike a naive direct computation of
      the gradient with respect to each
      weight individually. This efficiency
      makes it feasible to use gradient
      methods for training multilayer
      networks, updating weights to minimize
      loss; gradient descent, or variants
      such as stochastic gradient descent,
      are commonly used. The backpropagation
      algorithm works by computing the
      gradient of the loss function with
      respect to each weight by the chain
      rule, computing the gradient one layer
      at a time, iterating backward from the
      last layer to avoid redundant
      calculations of intermediate terms in
      the chain rule; this is an example of
      dynamic programming.[3]
      """

      # Wordsworth
      lucy = """
      She dwelt among the untrodden ways
      Beside the springs of Dove,
      A Maid whom there were none to praise
      And very few to love:

      A violet by a mossy stone
      Half hidden from the eye!
      —Fair as a star, when only one
      Is shining in the sky.

      She lived unknown, and few could know
      When Lucy ceased to be;
      But she is in her grave, and, oh,
      The difference to me!

      """

      # Davies
      full_of_cares = """
      What is this life if, full of care,
      We have no time to stand and stare.

      No time to stand beneath the boughs
      And stare as long as sheep or cows.

      No time to see, when woods we pass,
      Where squirrels hide their nuts in grass.

      No time to see, in broad daylight,
      Streams full of stars, like skies at night.

      No time to turn at Beauty's glance,
      And watch her feet, how they can dance.

      No time to wait till her mouth can
      Enrich that smile her eyes began.

      A poor life this if, full of care,
      We have no time to stand and stare.

      
      """

      # prepare a list with the data
      sentences = [
      jabberwocky, best_of_times, last_dream, mystery, mark_twain_dog, einstein,
      tweedledee, goldens_1, goldens_2, goldens_3, dog_soul, keats, attention, backprop, lucy, full_of_cares
      ]

 

      # return the list of sentences.
      def return_list(self):
            return self.sentences

def test_classes():
      #initialize class object
      scm = SearchCorpusModule()
      search_corpus  = scm.return_list()
      print(f"search corpus is : {search_corpus}")

if __name__ == "__main__":
      test_classes()