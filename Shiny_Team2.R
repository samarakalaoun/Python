library(dplyr)
library(stringr)
library(tidytext)
library(Rcpp)
library(plotly)
library(shiny)
library(shinydashboard)
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

## User Interface
ui <- dashboardPage(skin = "blue",
    dashboardHeader(title = "Survey Analysis"),
    
    dashboardSidebar(
        sidebarMenu(
            
            menuItem("Web", tabName = "web"),
            menuItem("Podcast", tabName = "podcast"),
            menuItem("App", tabName = "app"),
            menuItem("Cluster", tabName = "cluster")
        )
    ),
    
    dashboardBody(
        tabItems(
            tabItem(tabName = "web",
                    h2("Overview"),
                    tabBox(
                        id = "tabset1", height = "770px", width = "500px",
                        tabPanel("Intro",
                                 fluidRow(
                                     splitLayout(cellWidths = c("70%", "50%"),
                                                 column(width= 11, height = 20,
                                                        box(
                                                            title = "Word Cloud", width = "100%", height = "500px", solidHeader = TRUE, status = "primary",
                                                            wordcloud2Output("over_cloud", width = "100%", height = "400px")
                                                            )),
                                                 column(width = 9,
                                                        box(
                                                            sliderInput("overview_slider",
                                                            "Choose:",
                                                            min = .1,
                                                            max = .6,
                                                            value = .1)
                                                            ))
                                                 )
                                                 )),
                        tabPanel("Spider Graph",
                                 fluidRow(
                                     splitLayout(cellWidths = "90%",
                                                 column(width= 12, height = 30,
                                                        box(
                                                            title = "Bigram Network", width = "80%", height = "500px", solidHeader = TRUE, status = "primary",
                                                            plotOutput("spider", width = "100%", height = "440px")
                                                            ))
                                                 )
                                                 )),
                        tabPanel("Daily News (Word Cloud)", 
                                 fluidRow(
                                     splitLayout(cellWidths = c("70%", "50%"),
                                                 column(width= 12, height = 30,
                                                        box(
                                                            title = "NRC Word Cloud", width = 30, height = "730px", solidHeader = TRUE, status = "primary",
                                                            plotOutput("nrc_cloud", width = "100%", height = "670px")
                                                            )),
                                                 column(width = 6,
                                                        box(
                                                            title = "Number of words", width = 10, solidHeader = FALSE, status = "primary",
                                                            sliderInput("nrc_slider",
                                                                        "Maximum:",
                                                                        min = 30,
                                                                        max = 200,
                                                                        value = 180)
                                                            ),
                                                        
                                                        box(
                                                            title = "Letter size", width = 10, solidHeader = FALSE, status = "primary",
                                                            sliderInput("nrc_slider_size",
                                                                        "Choose the size:",
                                                                        min= .2,
                                                                        max = 2,
                                                                        value = .6)
                                                            )))
                                         )),
                        
                        tabPanel("Digital vs Printed (Word Cloud)", 
                                 fluidRow(
                                     splitLayout(cellWidths = c("70%", "50%"), 
                                                 column(width= 12, height = 30,
                                                        box(
                                                            title = "NRC Word Cloud", width = 30, height = "800px", solidHeader = TRUE, status = "primary",
                                                            plotOutput("nrc_dig_cloud", width = "100%", height = "700px")
                                                            )),
                                                        
                                                 column(width = 6,
                                                        box(
                                                            title = "Number of words", width = 10, solidHeader = FALSE, status = "primary",
                                                            sliderInput("nrc_dig_slider",
                                                                        "Maximum:",
                                                                        min = 30,
                                                                        max = 200,
                                                                        value = 180)
                                                            ),
                                                            
                                                            box(
                                                                title = "Letter size", width = 10, solidHeader = FALSE, status = "primary",
                                                                sliderInput("nrc_dig_slider_size",
                                                                            "Choose the size:",
                                                                            min= .2,
                                                                            max = 2,
                                                                            value = .6)
                                                                )))
                                                 )),
                                                 
                        tabPanel("Digital vs Printed (Bigram)", 
                                 fluidRow(
                                     column(width= 9, 
                                            box(
                                                title = "Bigram", width = 5, solidHeader = TRUE, status = "primary",
                                                verbatimTextOutput("dig_bigram", placeholder = TRUE)
                                            )))
                        )
                        
                        
                    )
            ),
            
            tabItem(tabName = "podcast",
                    h2("Podcast overview"),
                    tabBox(
                        id = "tabset2", height = "650px", width = "500px",
                        tabPanel("Sentiment",
                                 fluidRow(
                                     splitLayout(cellWidths = c("70%", "50%"), 
                                                 column(width= 12, height = 15, 
                                                        box(
                                                            title = "NRC Cloud", width = 20, height = "570px", solidHeader = TRUE, status = "primary",
                                                            plotOutput("user_cloud", width = "100%", height = "500px")
                                                            )),
                                                 
                                                 column(width = 6,
                                                        box(
                                                            title = "Number of words", width = 10, solidHeader = FALSE, status = "primary",
                                                            sliderInput("user_slider",
                                                                        "Maximum:",
                                                                        min = 30,
                                                                        max = 200,
                                                                        value = 180)
                                                            ),
                                                        
                                                        box(
                                                            title = "Letter size", width = 10, solidHeader = FALSE, status = "primary",
                                                            sliderInput("user_slider_size",
                                                                        "Choose the size:",
                                                                        min= .2,
                                                                        max = 2,
                                                                        value = .6)
                                                            )))
                                                 )),
                        tabPanel("Representative Tokens",
                                 fluidRow(
                                     column(width = 11, 
                                            box(
                                                title = "TF_IDF Question 3", width = 30, height = 15, solidHeader = TRUE, status = "primary",
                                                plotOutput("tf_idf", width = "100%", height = "400px")
                                            ))
                                 )),
                        tabPanel("Bigram", 
                                 fluidRow(
                                     column(width= 8, 
                                            box(
                                                title = "Bigram", width = 6, solidHeader = TRUE, status = "primary",
                                                verbatimTextOutput("user_bigram", placeholder = TRUE)
                                            )))
                        )
                    )),
            
            
            
            tabItem(tabName = "app",
                    h2("App"),
                    tabBox(
                        id = "tabset3", height = "650px", width = "500px",
                        
                        tabPanel("App Overview",
                                 fluidRow(
                                     splitLayout(cellWidths = c("70%", "50%"), 
                                                 column(width= 12, height = 30,
                                                        box(
                                                            title = "NRC Word Cloud", width = 20, height = "570px", solidHeader = TRUE, status = "primary",
                                                            plotOutput("app_cloud", width = "100%", height = "500px")
                                                            )),
                                                 
                                                column(width = 6,
                                                        box(
                                                            title = "Number of words", width = 10, solidHeader = FALSE, status = "primary",
                                                            sliderInput("app_slider",
                                                                        "Maximum:",
                                                                        min = 30,
                                                                        max = 200,
                                                                        value = 180)
                                                            ),
                                                       
                                                       box(
                                                           title = "Letter size", width = 10, solidHeader = FALSE, status = "primary",
                                                           sliderInput("app_slider_size",
                                                                       "Choose the size:",
                                                                       min= .2,
                                                                       max = 2,
                                                                       value = .6)
                                                           )))
                                                )),
                        tabPanel("Bigram",
                                 fluidRow(
                                     column(width= 9, 
                                            box(
                                                title = "Bigram", width = 5, solidHeader = TRUE, status = "primary",
                                                verbatimTextOutput("app_bigram", placeholder = TRUE)
                                            )))
                                 
                        ),
                        tabPanel("Representative Tokens",
                                 fluidRow(
                                     column(width = 10, 
                                            box(
                                                title = "TF_IDF Question 4", width = 20, height = 15, solidHeader = TRUE, status = "primary",
                                                plotOutput("tf_idf_app", width = "100%", height = "400px")
                                            ))
                                 )
                            
                        )
                        
                    )),
            
            
            tabItem(tabName = "cluster",
                    h2("Clustering"),
                    tabBox(
                        id = "tabset4", height = "800px", width = "500px",
                        tabPanel("Topics",
                                 fluidRow(
                                     splitLayout(cellWidths = "70%", 
                                                 column(width= 12, height = 30,
                                                        box(
                                                            title = "Dirichlet Allocation", width = 20, height = "500px", solidHeader = TRUE, status = "primary",
                                                            plotOutput("cluster", width = "100%", height = "400px")
                                                            ))),
                                                        
                                                 column(width = 6,
                                                        box(
                                                            title = "Number of topics", width = 10, height = 3, solidHeader = FALSE, status = "primary",
                                                            sliderInput("k_cluster",
                                                                        "Choose:",
                                                                        min= 2,
                                                                        max = 4,
                                                                        value = 3)
                                                        ))
                                     ))
                        ))
            )))
                    
                                                     
                                                           
                        
    
## Server
server <- shinyServer(function(input, output, session) {
    
    
    
    output$tabset3Selected <- renderText({
        input$tabset3
    })
    
    output$tokenb1 <- renderPrint({
        bigram_counts1
    })
    
    output$tokenb2 <- renderPrint({
        bigram_counts2
    })
    
    output$tokenb3 <- renderPrint({
        bigram_counts3
    })
    
    output$tokenb4 <- renderPrint({
        bigram_counts4
    })
   
    
     output$tabset1Selected <- renderText({
         input$tabset1
     })
     
     output$nrc_cloud <- renderPlot({
         df_tokenized1 %>%
             inner_join(get_sentiments("nrc")) %>%  #nrc=all flavors
             count(word, sentiment, sort=TRUE) %>% #token per sentiment
             acast(word ~sentiment, value.var="n", fill=0) %>%
             comparison.cloud(colors = c("red", "blue"),
                              max.words= input$nrc_slider, scale=c(input$nrc_slider_size,input$nrc_slider_size),
                              fixed.asp=TRUE, title.size=1)
     })
     
     output$spider <- renderPlot({
         ggraph(bigram_graph_team2, layout = "fr") +
             geom_edge_link()+
             geom_node_point()+
             geom_node_text(aes(label=name), vjust =1, hjust=1)
     })
     
     
     
     output$over_cloud <- renderWordcloud2({
         wordcloud2(d, size = input$overview_slider, minSize = 0, gridSize =  0,
                    fontFamily = 'Segoe UI', fontWeight = 'bold',
                    color = 'random-dark', backgroundColor = "white",
                    minRotation = -pi/4, maxRotation = pi/4, shuffle = TRUE,
                    rotateRatio = 0.4, shape = 'circle', ellipticity = 0.65,
                    widgetsize = NULL, figPath = NULL, hoverFunction = NULL)
         
     })
     
     output$nrc_dig_cloud <- renderPlot({
         df_tokenized2 %>%
             inner_join(get_sentiments("nrc")) %>%  #nrc=all flavors
             count(word, sentiment, sort=TRUE) %>% #token per sentiment
             acast(word ~sentiment, value.var="n", fill=0) %>%
             comparison.cloud(colors = c("red", "blue"),
                              max.words= input$nrc_dig_slider, scale=c(input$nrc_dig_slider_size,input$nrc_slider_size),
                              fixed.asp=TRUE, title.size=1)
     })
     
     output$dig_bigram <- renderPrint({
         bigram_counts2 
         
     })
     
     output$tabset2Selected <- renderText({
         input$tabset2
     })
     
     output$user_cloud <- renderPlot({
         df_tokenized3 %>%
             inner_join(get_sentiments("nrc")) %>%  #nrc=all flavors
             count(word, sentiment, sort=TRUE) %>% #token per sentiment
             acast(word ~sentiment, value.var="n", fill=0) %>%
             comparison.cloud(colors = c("red", "blue"),
                              max.words= input$user_slider, scale=c(input$user_slider_size,input$user_slider_size),
                              fixed.asp=TRUE, title.size=1)
     })
     
     output$tf_idf <- renderPlot({
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
     })
     
     output$user_bigram <- renderPrint({
         bigram_counts3 
         
     })
     
     output$tabset3Selected <- renderText({
         input$tabset3
     })
     
     output$app_bigram <- renderPrint({
         bigram_counts4 
         
     })
     
     output$app_cloud <- renderPlot({
         df_tokenized4 %>%
             inner_join(get_sentiments("nrc")) %>%  #nrc=all flavors
             count(word, sentiment, sort=TRUE) %>% #token per sentiment
             acast(word ~sentiment, value.var="n", fill=0) %>%
             comparison.cloud(colors = c("red", "blue"),
                              max.words= input$app_slider, scale=c(input$app_slider_size,input$app_slider_size),
                              fixed.asp=TRUE, title.size=1)
     })
     
     output$cluster <- renderPlot({
         ap_lda <- LDA(tdm_tot, k=input$k_cluster, control=list(seed=123))
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
         
         
     })
     
     output$tf_idf_app <- renderPlot({
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
     })
     
     
     
    })


## Run the application 
shinyApp(ui, server)
