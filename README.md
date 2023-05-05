Download Link: https://assignmentchef.com/product/solved-ece324-assignment-5-subjective-objective-sentence-classification-using-word-vectors-and-nlp
<br>
<h1>1         Sentence Classification – Problem Definition</h1>

Natural language processing, as we have discussed it in class, can provide the ability to work with the meaning of written language. As an illustration of that, in this assignment we will build models that try to determine if a sentence is objective (a statement based on facts) or <em>subjective </em>(a statement based on opinions).

In class we have described the concept and method to convert words (and possibly groups of words) into a vector (also called an <em>embedding</em>) that represents the meaning of the word. In this assignment we will make use of word vectors that have already been created (actually, <em>trained</em>), and use them as the basis for the three classifiers that you will build. The word vectors will be brought into your program and used to convert each word into a vector.

When working from text input, we need introduce a little terminology from the NLP domain: each word is first <em>tokenized </em>– i.e. made into word <em>tokens</em>. The first step has some complexity – for example, “I’m” should be separated to “I” and “am”, while “Los Angeles” should be considered together as a single word/token. After tokenization each word is converted into an identifying number (which is referred to both as its <em>index </em>or simply as a <em>word token</em>). With this index, the correct word vector can retrieved from a lookup table, which is referred to as the <em>embedding matrix</em>.

These indices are passed into different neural network models in this assignment to achieve the classification – subjective or objective – as illustrated below:

<table width="29">

 <tbody>

  <tr>

   <td width="29">32</td>

  </tr>

 </tbody>

</table>

“The fight scenes are fun”           Tokenize          4      427 453

Output

Text Sentence                                                  Discrete tokens                                      Probability

(Subjective)

Figure 1: High Level diagram of the Assignment 4 Classifiers for Subjective/Objective

Note that the first ‘layer’ of the neural network model will actually be the step that converts the index/token a word vector. (This could have been done on all of the training examples, but that would hugely increase the amount of memory required to store the examples). From there on, the neural network deals only the word vectors.

<h1>2         Setting Up Your Environment</h1>

<h2>2.1        Installing Libraries</h2>

In addition to PyTorch, we will be using two additional libraries:

<ul>

 <li><strong>torchtext </strong><a href="https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html">(</a><a href="https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html">https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutoria</a> <a href="https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html">html</a><a href="https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html">)</a>: This package consists of both data processing utilities and popular datasets for natural language, and is compatible with PyTorch. We will be using torchtext to process the text inputs into numerical inputs for our models.</li>

 <li><strong>SpaCy </strong><a href="https://spacy.io/">(</a><a href="https://spacy.io/">https://spacy.io/</a><a href="https://spacy.io/">)</a>: For ‘tokenizing’ English words. A text input is a sequence of symbols (letters, spaces, numbers, punctuation, etc.). The process of tokenization separates the text into units (such as words) that have linguistic significance, as described above in Section 1.</li>

</ul>

Install these two packages using the following commands:

pip install torchtext spacy python -m spacy download en

<h2>2.2        Dataset</h2>

We will use the Subjectivity dataset [2], introduced in the paper by Pang and Lee [5]. The data comes from portions of movie reviews from Rotten Tomatoes [3] (which are assumed <em>all </em>be subjective) and summaries of the plot of movies from the Internet Movie Database (IMDB) [1] (which are assumed <em>all </em>be objective). This approach to labeling the training data as objective and subjective may not be strictly correct, but will work for our purposes.

<h1>3         Preparing the data</h1>

<h2>3.1        Create train/validation/test splits</h2>

The data for this assignment was provided in the file you downloaded from Quercus. It contains the file data.tsv, which is a <em>tab</em>-separated-value (TSV) file. It contains 2 columns, text and label. The text column contains a text string (including punctuation) for each sentence (or fragment or multiple sentences) that is a data sample. The label column contains a binary value {0,1}, where 0 represents the objective class and 1 represents the subjective class.

As discussed in class, we will now use proper data separation, dividing the available data into <em>three </em>datasets: training, validation and <em>test</em>. Write a Python script split data.py to split the data.tsv into 3 files:

<ol>

 <li>tsv: this file should contain 64% of the total data</li>

 <li>tsv: this file should contain 16% of the total data</li>

 <li>tsv: this file should contain 20% of the total data</li>

</ol>

In addition, it is crucial to <strong>make sure that there are equal number of examples between the two classes </strong>in each of the train, validation, and test set, <strong>and have your script print out the number in each, and provide those numbers in your report</strong>.

Finally, created a <em>fourth </em>dataset, called overfit.tsv also with equal class representation, that contains only 50 training examples for use in debugging your models below.

<h2>3.2        Process the input data</h2>

The torchtext library is very useful for handling natural language text; we will provide the basic processing code to bring in the dataset and prepare it to be converted into word vectors. If you wish to learn more detail on this, the following tutorial the includes example uses of the library: <a href="https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8">https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8</a><a href="https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8">.</a> The code described in this section is already present in the skeleton code file main.py.

Below is a description of that code in the skeleton main.py that preprocesses the data:

<ol>

 <li>The Field object tells torchtext how each column in the TSV file will be processed when passed into the TabularDataset object. The following code instantiates two torchtext.data.Field objects, one for the “text” (sentences) and one for the“label” columns of the TSV data:</li>

</ol>

TEXT = data.Field(sequential=True,lower=True, tokenize=’spacy’, include_lengths=True)

LABELS = data.Field(sequential=False, use_vocab=False)

Details: <a href="https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.Field">https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.Field</a>

<ol start="2">

 <li>Next we load the train, validation, and test datasets to become datasets as was done in the previous assignments, with the torchtext method TabularDataset.splits, that is designed specifically for text input. main.py uses the following code, which assumes that the tsv files are in the folder data:</li>

</ol>

train_data, val_data, test_data = data.TabularDataset.splits(

path=’data/’, train=’train.tsv’, validation=’validation.tsv’, test=’test.tsv’, format=’tsv’, skip_header=True,

fields=[(’text’, TEXT), (’label’, LABELS)])

Details: <a href="https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.TabularDataset">https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.TabularDataset</a>

<ol start="3">

 <li>Next we need to create an object that can be <em>enumerated </em>(Python-style) to be used in the training loops – these are the objects that produce each batch in the training loop. The objects in each batch are accessed using the .text field and the .label field that was specified in the above line.</li>

</ol>

The iterator for the train/validation/test splits created earlier is done using the data.BucketIterator as shown below. This class will ensure that, within a batch, the size of the sentences will be as similar as possible, to avoid as much padding of the sentences as possible.

train_iter, val_iter, test_iter = data.BucketIterator.splits((train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),

sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

<ol start="4">

 <li>The Vocab object will contain the index (also called word token) for each unique word in the data set. The is done using the build vocab function, which looks through all of the given sentences in the data:</li>

</ol>

TEXT.build_vocab(train_data,val_data, test_data)

Details: <a href="https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.Field.build_vocab">https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.Fie</a>ld. <a href="https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.Field.build_vocab">build_vocab</a>

<h1>4         Baseline Model and Training</h1>

In your models.py file, you will first implement and train the <em>baseline </em>model (given below), which was discussed in class. Some of the code below will be re-usable for the other two models.

<h2>4.1        Loading GloVe Vector and Using Embedding Layer</h2>

As mentioned in Section 1, we will make use of word vectors that have already been created/trained. We will use the GloVe [6] pre-trained word vectors in an “embedding layer” (which is just that “lookup matrix” described earlier) in PyTorch in two steps:

<ol>

 <li>(As given in the skeleton file py code) Using the vocab object from Section 3.2, item number 4, download (the first time this is run) and load the vectors that are downloaded into the vocab object, as follows:</li>

</ol>

TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name=’6B’, dim=100)) vocab = TEXT.vocab

You can see the shape of the complete set of word vectors by printing out the the shape of the vectors object as follows, which will be the number of unique words in all the training sets and the embedding dimension (word vector size). print(“Shape of Vocab:”,TEXT.vocab.vectors.shape)

This loads word vectors into a GloVe class (see documentation <a href="https://torchtext.readthedocs.io/en/latest/vocab.html#torchtext.vocab.GloVe">https://torchtext.readthedo</a>cs. <a href="https://torchtext.readthedocs.io/en/latest/vocab.html#torchtext.vocab.GloVe">io/en/latest/vocab.html#torchtext.vocab.GloVe</a><a href="https://torchtext.readthedocs.io/en/latest/vocab.html#torchtext.vocab.GloVe">)</a> This GloVe model was trained with six billion words to produce a word vector size of 100, as described in class. This will download a rather large <strong>862 MB </strong>zip file into the folder named .vector cache, which might take some time; this file expands into a 3.3Gbyte set of files, but you will only need one of those files, labelled glove.6B.100d.txt, and so you can delete the rest (but don’t delete the file glove.6B.100d.txt.pt that will be created by main.py, which is the binary form of the vectors). Note that .vector cache folder, because it starts with a ‘.’, is typically not a visible folder, and you’ll have to make it visible with an operating system-specific view command of some kind. (<a href="https://support.microsoft.com/en-ca/help/14201/windows-show-hidden-files">Windows</a><a href="https://support.microsoft.com/en-ca/help/14201/windows-show-hidden-files">,</a> <a href="https://apple.stackexchange.com/questions/309450/how-to-unhide-files-on-mac">Mac</a><a href="https://apple.stackexchange.com/questions/309450/how-to-unhide-files-on-mac">)</a> Once downloaded your code can now access the vocabulary object within the text field object by calling .vocab attribute on the text field object.

<ol start="2">

 <li>The step that converts the input words from an index number (a word token) into the word vector is actually done inside the nn.module model class. So, when defining the layers in your model class, you must add an embedding layer with the function Embedding.from pretrained, and pass in vocab.vectors as the argument where vocab is the Vocab object. The code for this is shown below in the model section, and is given in the skeleton file models.py.</li>

</ol>

Details: <a href="https://pytorch.org/docs/stable/nn.html?highlight=from_pretrained#torch.nn.Embedding.from_pretrained">https://pytorch.org/docs/stable/nn.html?highlight=from_pretrained#torch</a>.

<a href="https://pytorch.org/docs/stable/nn.html?highlight=from_pretrained#torch.nn.Embedding.from_pretrained">nn.Embedding.from_pretrained</a>

<h2>4.2        Baseline Model</h2>

The     fight    scenes are        fun

Figure 2: A simple baseline architecture

The <em>baseline </em>model was discussed in class and is illustrated in Figure 2. It first converts each of the word tokens into a vector using the GloVe word embeddings that were downloaded. It then computes the average of those word embeddings in a given sentence. The idea is that this becomes the ’average’ meaning of the entire sentence. This is fed to a fully connected layer which produces a scalar output with sigmoid activation (which is computed inside the BCEWithLogitsLoss losss function) to represent the probability that the sentence is in the subjective class.

The code for this Baseline class is given below, and is also provided in the skeleton file models.py. Read it and make sure you understand it.

class Baseline(nn.Module):

def __init__(self, embedding_dim, vocab):

super(Baseline, self).__init__()

self.embedding = nn.Embedding.from_pretrained(vocab.vectors) self.fc = nn.Linear(embedding_dim, 1)

def forward(self, x, lengths=None): #x has shape [sentence length, batch size] embedded = self.embedding(x) average = embedded.mean(0) # [sent len, batch size, emb dim] output = self.fc(average).squeeze(1)

# Note – using the BCEWithLogitsLoss loss function

# performs the sigmoid function *as well* as well as

# the binary cross entropy loss computation # (these are combined for numerical stability) return output

<h2>4.3        Training the Baseline Model</h2>

In main.py write a training loop to iterate through the training dataset and train the baseline model. Use the hyperparameters given in Table 1. Note that we have not used the Adam optimizer yet in this course; it will be discussed in a later lecture. The Adam optimizer is invoked the same way as the SGD optimizer, using optim.Adam.

<table width="295">

 <tbody>

  <tr>

   <td width="137"><strong>Hyperparameter</strong></td>

   <td width="158"><strong>Value</strong></td>

  </tr>

  <tr>

   <td width="137">Optimizer</td>

   <td width="158">Adam</td>

  </tr>

  <tr>

   <td width="137">Learning Rate</td>

   <td width="158">0.001</td>

  </tr>

  <tr>

   <td width="137">Batch Size</td>

   <td width="158">64</td>

  </tr>

  <tr>

   <td width="137">Number of Epochs</td>

   <td width="158">25</td>

  </tr>

  <tr>

   <td width="137">Loss Function</td>

   <td width="158">BCEWithLogitsLoss()</td>

  </tr>

 </tbody>

</table>

Table 1: Hyperparameters to Use in Training the Models

The objects train_iter, val_iter, test_iter and overfit_iter described in Section 3.2 are the iterable objects that will produce the batches of batch_size in each training inner loop step. The torchtext.data.batch.Batch object is given by the iterator, from which you can obtain both the text input and the length of the sentence sequences from the .text field of the Batch object, as follows, assuming that batch is the object returned from the iterator: batch_input, batch_input_length = batch.text

Where batch_input is the set of text sentences in the batch.

The details on this object can be found in <a href="https://spacy.io/usage/spacy-101#annotations-token">https://spacy.io/usage/spacy-101#annotations-token</a><a href="https://spacy.io/usage/spacy-101#annotations-token">.</a>

<h2>4.4        Overfitting to debug</h2>

As was done in Assignment 3, debug your model by using <em>only </em>the very small overfit.tsv set (described above, which you’ll have to turn into a dataset and iterator as shown in the given code), and see if you can <em>overfit </em>your model and reach a much higher training accuracy than validation accuracy. (The baseline model won’t have enough parameters that you can get an accuracy of 100%; the cnn and rrnn models will have enough). You will need more than 25 epochs to succeed in overfitting. Recall that the purpose of doing this is to be able to make sure that the input processing and output measurement is working.

It is also recommended that you include some useful logging in the loop to help you keep track of progress, and help in debugging.

Provide the training loss and accuracy plot for the overfit data in your Report.

<h2>4.5        Full Training Data</h2>

Once you’ve succeeded in overfitting the model, then use the full training dataset to train your model, using the hyper-parameters given in Table 1.

In main.py write an evaluation loop to iterate through the validation dataset to evaluate your model. It is recommended that you call the evaluation function in the training loop (perhaps every epoch or two) to make sure your model isn’t overfitting. Keep in mind if you call the evaluation function too often, it will slow down training.

Give the training and validation loss and accuracy curves vs. epoch in your <strong>Report</strong>, and report the final test accuracy. Evaluate the test data and provide the accuracy result in your <strong>Report</strong>.

<h2>4.6        Saving and loading your model</h2>

In main.py, save the model with the lowest validation error with torch.save(model, ’model baseline.pt’). You will need to load this files in the next section. See <a href="https://pytorch.org/tutorials/beginner/saving_loading_models.html">https://pytorch.org/tutorials/ </a><a href="https://pytorch.org/tutorials/beginner/saving_loading_models.html">beginner/saving_loading_models.html</a> for detail on saving and loading.

<h1>5         Convolutional Neural Network (CNN)</h1>

Embedding Dim

Figure 3: A convolutional neural network architecture

The second architecture, described in class and illustrated in Figure 3, is to use a CNN-based architecture that is inspired by Yoon et al. [4]. Yoon first proposed using CNNs in the context of NLP. You will write the code for the CNN model class in models.py file with the following specifications:

<ol>

 <li>Group together all the vectors of the words in a sentence – the vectors of those words – to form a embedding dim * N matrix, where N is the number of words (and therefore tokens) in the sentence. Different sentences will have different lengths, which is unusal for a CNN but that will be dealt with in the final pooling step. Note that embedding dim is the size of the word vector, 100.</li>

 <li>The architecture consists of two convolutional layers that both operate on the word vector group created above, but with different kernel sizes. The kernel sizes are [k, embedding dim], and you should use he following values for <em>k </em>∈ 2<em>,</em> Use 50 kernels for each of the two kernel sizes. Note that this organization of convoluational layers is different from your prior use of CNNs in which one layer fed into the next; these are operating on the same input. Note also, that, even though the kernel sizes span the entire embedding dimension, you can still use the nn.conv2d method, and explicitly specify the size of a kernel using the kernel_size=(kx,ky) notation.</li>

 <li>Use the ReLU activation function on the convolution output.</li>

 <li>To handle the variable sentence lengths, we perform a MaxPool operation on the convolution layer feature output (after activation), along the sentence length dimension. That is, we compute the maximum across the entire sentence length, and get one output feature/number from each sentence for each kernel.</li>

 <li>Concatenate the outputs from the maxpool operations above to form a fixed length vector of dimension 100 – because each of the two kernel sizes is used 50 times each in the two different conv layers.</li>

 <li>Finally, similar to the baseline architecture, use a fully connected layer to a scalar output with sigmoid activation to represent the probability that the sentence is in the subjective class. (Recall that the BCEwithLogitsLoss function computes the sigmoid as was the loss; to actually determine the probability when printing out an answer, you’ll need to separately apply a sigmoid on the neural network output value.</li>

</ol>

<h2>5.1        Overfit, Training and Test</h2>

Once you’ve created the code for the CNN model, follow the same processes described in Sections

4.3, 4.4, 4.5 and 4.6, except change the file name of the saved model to be model_cnn.

<h1>6         Recurrent Neural Network (RNN)</h1>

Figure 4: A Recurrent Neural Network Architecture

The third architecture, illustrated in Figure 4, is to use a Recurrent Neural Network (RNN)-based architecture. A recurrent neural network has a hidden state <em>h</em><sub>0 </sub>that is initialized at the start of a sequence of inputs (typically to all zeroes). The network takes in the input – the vector corresponding to a word in the sentence, <em>x<sub>t </sub></em>at each ‘step’ of the sequence, as illustrated in Figure 4. It computes the new hidden state as a function of the previous hidden state and input the word vector. In this way the newly computed hidden state retains information from the previous inputs and hidden states, as expressed in this equation:

<em>h<sub>t </sub></em>= <em>f</em>(<em>h<sub>t</sub></em>−<sub>1</sub><em>,x<sub>t</sub></em>)                                                          (1)

The final hidden state (<em>h<sub>T</sub></em>, where <em>T </em>is the number of words in the sentence), is produced after the full sequence of words is processed, and <em>is a representation of the sentence </em>just as the average produced in the baseline above is a representation of the sentence. Similar to the baseline, we then use a fully connected layer to generate a single number output, together with sigmoid activation to produce the probability that the sentence is in the subjective class.

Here are some guidelines to help you implement the RNN model in models.py file:

<ol>

 <li>You should use the Gated Recurrent Unit (GRU) as the basic RNN cell, which will essentially fulfill the function of the blue boxes in Figure 4. The GRU takes in the hidden state and the input, and produces a new hidden state. In the init function of your RNN model, use the GRU cell (from Pytorch). Set the <strong>embedding dimension to be 100 </strong>(as that is the size of the word vectors) and select the <strong>hidden dimension to be 100</strong>.</li>

 <li>As usual, during training, we send <em>batches </em>of sentences through the network at one time, with just one call to the model forward function. One issue with this is that the sentence lengths will differ within one batch. The shorter sentences are padded from the end onward to the longest sentence length in the batch. PyTorch’s GRU module can take in a batch of several sentences and return the hidden states for all of the (words × batch size) in one call without a for-loop, as well as the last hidden states (which is the one we use to generate the answer).</li>

</ol>

However, there is a problem if you simply use the last hidden states returned: for the shorter setences, these will be the wrong hidden states, because the sentence ended earlier, as shown on the left side of Figure 5. Instead, you can use PyTorch’s nn.utils.rnn.pack padded sequence function (see documentation <a href="https://pytorch.org/docs/stable/nn.html?highlight=pack_padded_sequence#torch.nn.utils.rnn.pack_padded_sequence">https://pytorch.org/docs/stable/nn.html?highlight=pack</a>_ <a href="https://pytorch.org/docs/stable/nn.html?highlight=pack_padded_sequence#torch.nn.utils.rnn.pack_padded_sequence">padded_sequence#torch.nn.utils.rnn.pack_padded_sequence</a><a href="https://pytorch.org/docs/stable/nn.html?highlight=pack_padded_sequence#torch.nn.utils.rnn.pack_padded_sequence">)</a> to pack the word embeddings in the batch together and run the RNN on this object. The resulting final hidden state from the RNN will be the correct final hidden state for each sentence (see Figure 5 (Right)), not simply the hidden state at the maximum length sentence for all sentences.

<h2>6.1        Overfit, Training and Test</h2>

Once you’ve created the code for the RNN model, follow the same processes described in Sections

4.3, 4.4, 4.5 and 4.6, except change the file name of the saved model to be model_rnn.

<h1>7         Testing on Your Own Sentence</h1>

In this section, you will write a Python script subjective bot.py that prompts the user for a sentence input on the command line, and prints the classification from each of the three models, as well as the probability that this sentence is subjective. This was demonstrated in class, in Lecture

<ol start="18">

 <li>Specifically, running the command python subjective bot.py will:</li>

</ol>

sentence length                                                        sentence length

Figure 5: In a batch, each square represents the RNN hidden state at different words in the sequence (the columns) and input example in the batch (the rows). The shaded cells represent that there is an input token at that word in the sequence, while a white filled cell indicates a padded input (with a word vector containing zeroes). <strong>Left</strong>: If we naively took the hidden at the last column (at the maximum time step), then we will be getting the hidden state when the RNN has been fed padding vectors (zeroes). <strong>Right</strong>: We want to get the RNN hidden state at the last word for each sequence.

<ol>

 <li>Print “Enter a sentence” to the console, then on the next line, the user can type in a sentence string (with punctuations, etc.). (Hint: you can use the built-in input() function; also use import readline when you do that.)</li>

 <li>For each model trained, print to the console a string in the form of “Model [baseline|rnn|cnn]: [subjective|objective] (x.xxx)”, where xxx is the prediction probability that the sentence is subjective in the range [0,1] up to 3 decimal places.</li>

 <li>Print “Enter a sentence” prompt again, in an infinite loop until the user decides to terminate the Python program.</li>

</ol>

An example output on the console is given below:

Enter a sentence

What once seemed creepy now just seems campy

Model baseline: subjective (0.964)

Model rnn: subjective (0.999)

Model cnn: subjective (1.000)

Enter a sentence

The script can be broken down into several steps that you should implement:

<ol>

 <li>Obtain the Vocab object by performing the same preprocessing that was done in Part 3. This object will convert the string tokens into integer.</li>

 <li>Load the saved parameters for models you’ve trained: model = torch.load(’filename.pt’)</li>

 <li>You’ll need a tokenizer function, which takes in a string input and converts the words to tokens using the SpaCy as follows:</li>

</ol>

import spacy

def tokenizer(text):

spacy_en = spacy.load(’en’)

return [tok.text for tok in spacy_en(text)]

To convert the sentence string that has been input:

tokens = tokenizer(sentence)

Details: <a href="https://spacy.io/usage/spacy-101#annotations-token">https://spacy.io/usage/spacy-101#annotations-token</a>

<ol start="4">

 <li>To convert each string token to an integer, use the .stoi variable, which is a dictionary with string as the key and integer as the value, in the Vocab object as done here:</li>

</ol>

token_ints = [vocab.stoi[tok] for tok in tokens]

Details: <a href="https://torchtext.readthedocs.io/en/latest/vocab.html#torchtext.vocab.Vocab">https://torchtext.readthedocs.io/en/latest/vocab.html#torchtext.vocab. </a><a href="https://torchtext.readthedocs.io/en/latest/vocab.html#torchtext.vocab.Vocab">Vocab</a>

<ol start="5">

 <li>Convert the list of token integers into a LongTensor with the shape [L,1], where L is the number of tokens: token_tensor = torch.LongTensor(token_ints).view(-1,1) # Shape is [sentence_len, 1]</li>

 <li>Create a tensor for the length of the sentence with the shape [1]:</li>

</ol>

lengths = torch.Tensor([len(token_ints)])

This will be needed when calling the RNN model.

<ol start="7">

 <li>You can convert the torch Tensor into a numpy array by calling .detach().numpy() on the torch tensor object before printing so that the print formatting will match the examples.</li>

</ol>

<h1>8         Experimental and Conceptual Questions</h1>

<ol>

 <li>After training on the three models, report the loss and accuracy on the train/validation/test in a total. There should be a total of 18 numbers. Which model performed the best? Is there a significant difference between the validation and test accuracy? Provide a reason for your answer.</li>

 <li>In the baseline model, what information contained in the original sentence is being ignored? How will the performance of the baseline model inform you about the importance of that information?</li>

 <li>For the RNN architecture, examine the effect of using pack padded sequence to ensure that we did indeed get the correct last hidden state (Figure 5 (Right)). Train the RNN and report the loss and accuracy on the train/validation/test under these 3 scenarios:

  <ul>

   <li>Default scenario, with using pack padded sequence and using the BucketIterator</li>

   <li>Without calling pack padded sequence, and using the BucketIterator</li>

   <li>Without calling pack padded sequence, and using the Iterator. What do you notice about the lengths of the sentences in the batch when using Iterator class instead?</li>

  </ul></li>

</ol>

Given the results of the experiment, explain how you think these two factors affect the performance and why.

<ol start="4">

 <li>In the CNN architecture, what do you think the kernels are learning to detect? When performing max-pooling on the output of the convolutions, what kind of information is the model discarding? Compare how this is different or similar to the baseline model’s discarding of information.</li>

 <li>Try running the subjective py script on 4 sentences that you come up with yourself, where 2 are definitely objective/subjective, while 2 are borderline subjective/objective, according to your opinion. <strong>Include your console output in the write up</strong>. Comment on how the three models performed and whether they are behaving as you expected. Do they agree with each other? Does the majority vote of the models lead to correct answer for the 4 cases? Which model seems to be performing the best?</li>

 <li>Describe your experience with Assignment 4:

  <ul>

   <li>How much time did you spend on Assignment 4?</li>

   <li>What did you find challenging?</li>

   <li>What did you enjoy?</li>

   <li>What did you find confusing?</li>

   <li>What was helpful?</li>

  </ul></li>

</ol>