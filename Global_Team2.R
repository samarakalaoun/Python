library(dplyr)
library(stringr)
library(tidytext)
library(Rcpp)
library(plotly)
library(pdftools)
library(shapeR)
library(tidyverse)
library(textshape)
library(textreadr)
library(tidytext)
library(tidyr)
library(scales)
library(ggplot2)
library(textdata)
library(wordcloud)
library(wordcloud2)
library(RColorBrewer)
library(tm)
library(NLP)
library(topicmodels)
library(reshape2)
library(quanteda)
library(readr)
library(ggraph)
library(igraph)


## Importing document of answers and questions
Qs <- read_document(file="Questions_T2.docx")
Ans <- read_document(file="Answers_T2.docx")
class_combo <- c(Qs, Ans)

my_df <- as.data.frame(matrix(nrow=32, ncol=4))
for(z in 1:4){
  for(i in 1:32){
    my_df[i,z]<- Ans[i*4+z-4]
  }#closing z loop
}#closing i loop


##########################################################
#SPLITING THE DF FOR EACH QUESTION
##########################################################

my_ans1 <- my_df$V1  #spliting for Q1
#print(my_ans1)
df1 <- data_frame(line=1:32, text=my_ans1)

my_ans2 <- my_df$V2  #spliting for Q2
#print(my_ans2)
df2 <- data_frame(line=1:32, text=my_ans2)

my_ans3 <- my_df$V3  #spliting for Q3
#print(my_ans3)
df3 <- data_frame(line=1:32, text=my_ans3)

my_ans4 <- my_df$V4  #spliting for Q4
#print(my_ans4)
df4 <- data_frame(line=1:32, text=my_ans4)


##########################################################
#TOKEN TOKEN TOKENNNNN
##########################################################


data(stop_words)
df_tokenized1 <- df1 %>%
  unnest_tokens(word, text) %>% #error while Tokenizing
  anti_join(stop_words) %>%
  count(word, sort = TRUE)

#words <- tweets_words %>% count(word, sort=TRUE)

#data(stop_words)
custom_stop_Q2 <- tibble(word=c("it's", "they're"), lexicon = rep('cust'), each=2) 

df_tokenized2 <- df2 %>%
  unnest_tokens(word, text) %>% 
  anti_join(stop_words) %>%
  anti_join(custom_stop_Q2) %>%
  count(word, sort = TRUE)

#data(stop_words)
custom_stop_Q3 <- tibble(word=c("i", "to", "the", "and", "are", "but", "they", "it", "that", "what"), lexicon = rep('cust'), each=10)
df_tokenized3 <- df3 %>%
  unnest_tokens(word, text) %>% #error while Tokenizing
  anti_join(custom_stop_Q3) %>%
  count(word, sort = TRUE)


#data(stop_words)
custom_stop_Q4 <- tibble(word=c("i", "to", "the","an", "and", "are","my","for","me","a","if", "but", "they", "it", "that", "what"), lexicon = rep('cust'), each=16)
df_tokenized4 <- df4 %>%
  unnest_tokens(word, text) %>% #error while Tokenizing
  anti_join(custom_stop_Q4) %>%
  count(word, sort = TRUE)


###################################################
#### Adding positive and negative sentiments ######
###################################################


df_tokenized1 %>%
  inner_join(get_sentiments("nrc")) %>%  #nrc=all flavors
  count(word, sentiment, sort=TRUE) %>% #token per sentiment
  acast(word ~sentiment, value.var="n", fill=0) %>%
  comparison.cloud(colors = c("red", "blue"),
                   max.words=200, scale=c(.6,.6),
                   fixed.asp=TRUE, title.size=1)

df_tokenized2 %>%
  inner_join(get_sentiments("nrc")) %>%  #nrc=all flavors
  count(word, sentiment, sort=TRUE) %>% #token per sentiment
  acast(word ~sentiment, value.var="n", fill=0) %>%
  comparison.cloud(colors = c("red", "blue"),
                   max.words=200, scale=c(.6,.6),
                   fixed.asp=TRUE, title.size=1)

df_tokenized3 %>%
  inner_join(get_sentiments("nrc")) %>%  #nrc=all flavors
  count(word, sentiment, sort=TRUE) %>% #token per sentiment
  acast(word ~sentiment, value.var="n", fill=0) %>%
  comparison.cloud(colors = c("red", "blue"),
                   max.words=200, scale=c(.6,.6),
                   fixed.asp=TRUE, title.size=1)

df_tokenized4 %>%
  inner_join(get_sentiments("nrc")) %>%  #nrc=all flavors
  count(word, sentiment, sort=TRUE) %>% #token per sentiment
  acast(word ~sentiment, value.var="n", fill=0) %>%
  comparison.cloud(colors = c("red", "blue"),
                   max.words=200, scale=c(.6,.6),
                   fixed.asp=TRUE, title.size=1)


#######
#################### Applying class
#######

df_tokenized1%>% #which book did you select?
  inner_join(get_sentiments("nrc")) %>%
  count(word, sentiment, sort=T) %>%
  ungroup()

df_tokenized2%>% #which book did you select?
  inner_join(get_sentiments("nrc")) %>%
  count(word, sentiment, sort=T) %>%
  ungroup()

df_tokenized3%>% #which book did you select?
  inner_join(get_sentiments("nrc")) %>%
  count(word, sentiment, sort=T) %>%
  ungroup()

df_tokenized4%>% #which book did you select?
  inner_join(get_sentiments("nrc")) %>%
  count(word, sentiment, sort=T) %>%
  ungroup()

###Combining the questions to one matrix
my_combined <- bind_rows(
  mutate(df_tokenized1, location = "Q1"),
  mutate(df_tokenized2, location = "Q2"),
  mutate(df_tokenized3, location = "Q3"),
  mutate(df_tokenized4, location = "Q4")
)


############################################
## GLOBAL CLOUD 
############################################

#Turning to corpus 
corp <- Corpus(VectorSource(my_combined))
corp <- tm_map(corp, content_transformer(tolower))
corp <- tm_map(corp, removeNumbers)
corp <- tm_map(corp, removeWords, stopwords("english"))
corp <- tm_map(corp, removeWords, c('don<U+0092>t',  'q1', 'q2', 'q3', 'q4'))
corp <- tm_map(corp, removePunctuation)
corp <- tm_map(corp, stripWhitespace)



#Building a TDM  
dtm <- TermDocumentMatrix(corp)
m <- as.matrix(dtm)
v <- sort(rowSums(m), decreasing = TRUE)
d <- data.frame(word = names(v), freq=v)

d <- d[-c(1), ]

d <- d %>%
  mutate(word = str_extract(word, "[a-z']+")) %>%
  anti_join(stop_words)


head(d, 10)

wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

complete_wordc <- wordcloud2(d, size = .3, minSize = 0, gridSize =  0,
           fontFamily = 'Segoe UI', fontWeight = 'bold',
           color = 'random-dark', backgroundColor = "white",
           minRotation = -pi/4, maxRotation = pi/4, shuffle = TRUE,
           rotateRatio = 0.4, shape = 'circle', ellipticity = 0.65,
           widgetsize = NULL, figPath = NULL, hoverFunction = NULL)



###############################################################
## TF - IDF Analysis
##############################################################


#Addinf the TF-IDF columns
my_combined <- my_combined %>%
  bind_tf_idf(word, location, n)

#Ordering to see the most important words
my_combined %>%
  arrange(desc(tf_idf))


#############
# looking at the graphical apprach:


my_combined %>%
  arrange(desc(tf_idf)) %>%
  mutate(word=factor(word, levels=rev(unique(word)))) %>%
  group_by(location) %>%
  top_n(7) %>%
  ungroup %>%
  ggplot(aes(word, tf_idf, fill=location))+
  geom_col(show.legend=FALSE)+
  labs(x=NULL, y="tf-idf")+
  facet_wrap(~location, ncol=2, scales="free")+
  coord_flip()

#########
#Bigrams
#########

my_bigrams1 <- df1 %>%
  unnest_tokens(bigram, text, token = "ngrams", n=2)
my_bigrams1

my_bigrams2 <- df2 %>%
  unnest_tokens(bigram, text, token = "ngrams", n=2)
my_bigrams2

my_bigrams3 <- df3 %>%
  unnest_tokens(bigram, text, token = "ngrams", n=2)
my_bigrams3

my_bigrams4 <- df4 %>%
  unnest_tokens(bigram, text, token = "ngrams", n=2)
my_bigrams4

###################################
# Separating the words from bigrams
###################################



bigrams_separated1 <- my_bigrams1 %>%
  separate(bigram, c("word1", "word2"), sep = " ")
bigrams_separated1

bigrams_separated2 <- my_bigrams2 %>%
  separate(bigram, c("word1", "word2"), sep = " ")
bigrams_separated2

bigrams_separated3 <- my_bigrams3 %>%
  separate(bigram, c("word1", "word2"), sep = " ")
bigrams_separated3

bigrams_separated4 <- my_bigrams4 %>%
  separate(bigram, c("word1", "word2"), sep = " ")
bigrams_separated4

###################################
# Filtering the bigrams
###################################

bigrams_filtered1 <- bigrams_separated1 %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word)
bigrams_filtered1

bigrams_filtered2 <- bigrams_separated2 %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word)
bigrams_filtered2

bigrams_filtered3 <- bigrams_separated3 %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word)
bigrams_filtered3

bigrams_filtered4 <- bigrams_separated4 %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word)
bigrams_filtered4

##############################################
# Creating bigrams after removing "stop-words"
##############################################

bigram_counts1 <- bigrams_filtered1 %>%
  count(word1, word2, sort = TRUE)
bigram_counts1

bigram_counts2 <- bigrams_filtered2 %>%
  count(word1, word2, sort = TRUE)
bigram_counts2

bigram_counts3 <- bigrams_filtered3 %>%
  count(word1, word2, sort = TRUE)
bigram_counts3

bigram_counts4 <- bigrams_filtered4 %>%
  count(word1, word2, sort = TRUE)
bigram_counts4

######################################################
####### VISUALISING A BIGRAM NETWORK #################
######################################################



my_combined2 <- bind_rows(
  mutate(bigram_counts1, location = "Q1"),
  mutate(bigram_counts2, location = "Q2"),
  mutate(bigram_counts3, location = "Q3"),
  mutate(bigram_counts4, location = "Q4")
)


my_combined_counts <- bind_rows(
  mutate(bigram_counts1),
  mutate(bigram_counts2),
  mutate(bigram_counts3),
  mutate(bigram_counts4)
)



bigram_graph_team2 <- my_combined_counts %>%
  filter(n>2) %>%
  graph_from_data_frame()

bigram_graph_team2


ggraph(bigram_graph_team2, layout = "fr") +
  geom_edge_link()+
  geom_node_point()+
  geom_node_text(aes(label=name), vjust =1, hjust=1)



#######################################
#CREATING CORPUS AND TDM FOR OUR SURVEY
######################################


total_df<- my_df %>% 
  mutate(id = seq(1:nrow(my_df))) %>% 
  melt('id')

corpus_tot <- VCorpus(VectorSource(total_df$value)) 


#creating a Term-Document-Matrices OM.tdm <- TermDocumentMatrix(OMcorp,


tdm_tot <- DocumentTermMatrix(corpus_tot,
                              
                              control = 
                                list(removePunctuation = TRUE,
                                     stopwords = TRUE,
                                     tolower = TRUE))#,
# removeNumbers = TRUE,
# bounds = list(global = c(1, Inf)))) #only words that appear more than 3 times


#calling the Latent Dirichlet Allocation algorithm
ap_lda <- LDA(tdm_tot, k=3, control=list(seed=123))
ap_lda

#now we are looking for the per topic per word probabilities aka. beta
#beta - what is the probability that "this term" will be generated by "this topic"
ap_topics <- tidy(ap_lda, matrix="beta")
ap_topics


top_terms <- ap_topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)


#lets plot the term frequencies by topic
top_terms %>%
  mutate(term=reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend=FALSE) +
  facet_wrap(~topic, scales = "free") +
  coord_flip()

