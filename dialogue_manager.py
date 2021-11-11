import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from utils import *


class ThreadRanker(object):
    def __init__(self):
        #self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.word_embeddings, self.embeddings_dim = load_embeddings('starspace_embedding.tsv')
        #self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']
        self.thread_embeddings_folder = './thread_embeddings_by_tags'

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.
        
        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim) #### YOUR CODE HERE ####
        best_thread = pairwise_distances_argmin(question_vec.reshape(1,-1),thread_embeddings) #### YOUR CODE HERE ####
        
        return thread_ids[best_thread]


class DialogueManager(object):
    def __init__(self):
        print("Loading resources...")

        # Intent recognition:
        # self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        # self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])
        self.intent_recognizer =  pickle.load(open('./intent_recognizer.pkl', 'rb'))
        self.tfidf_vectorizer = pickle.load(open('./tfidf_vectorizer.pkl', 'rb'))

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        #self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.tag_classifier =  pickle.load(open('./tag_classifier.pkl', 'rb'))
        self.thread_ranker = ThreadRanker()
        self.__init_chitchat_bot()

    def __init_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # Create an instance of the ChatBot class.
        # Set a trainer set_trainer(ChatterBotCorpusTrainer) for the ChatBot.
        # Train the ChatBot with "chatterbot.corpus.english" param.
        # Note that we use chatterbot==0.7.6 in this project. 
        # You are welcome to experiment with other versions but they might have slightly different API.
        
        ########################
        #### YOUR CODE HERE ####
        ########################

        chatbot = ChatBot('ChitChat')
        trainer = ChatterBotCorpusTrainer(chatbot)
        trainer.train("chatterbot.corpus.english")
        trainer.train("chatterbot.corpus.english.conversations")
        self.chitchat_bot = chatbot

        # remove this when you're done
        # raise NotImplementedError(
        #     "Open dialogue_manager.py and fill with your code. In case of Google Colab, download"
        #     "(https://github.com/hse-aml/natural-language-processing/blob/master/project/dialogue_manager.py), "
        #     "edit locally and upload using '> arrow on the left edge' -> Files -> UPLOAD")
       
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        
        prepared_question = text_prepare(question) #### YOUR CODE HERE ####
        features = self.tfidf_vectorizer.transform([prepared_question]) #### YOUR CODE HERE ####
        intent = self.intent_recognizer.predict(features) #### YOUR CODE HERE ####

        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = self.chitchat_bot.get_response(question) #### YOUR CODE HERE ####
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)[0] #### YOUR CODE HERE ####
            
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(question, tag) #### YOUR CODE HERE ####
            
            return self.ANSWER_TEMPLATE % (tag, thread_id)
