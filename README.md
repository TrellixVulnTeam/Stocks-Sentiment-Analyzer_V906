# Stalk Sentiment Analyzer

[![Python Versions](https://img.shields.io/pypi/pyversions/yt2mp3.svg)](https://pypi.python.org/pypi/yt2mp3/)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)




<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#getting-started">Getting Started</a>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#acknowledgements">Literature References </a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This is a Streamlit deployed Web Application, used to analyze and compare stocks. Mass psychology's effects may be one of the factors driving the market. We can measure and use to predict market movement with surprising accuracy levels. We used the the social media and news contents for scrapping real-time information.

A list of commonly used resources that I find helpful are listed in the literature references.

### Built With

This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [Streamlit](https://streamlit.io/)
* [Flair](https://github.com/flairNLP/flair)
* [PyTorch](https://pytorch.org/)


<!-- GETTING STARTED -->
## Getting Started

We have used a pre-trained sentiment analysis model from the **flair**(DistilBERT Model) library. This model splits the text into character-level tokens and uses the DistilBERT model to make predictions of words, that the network has never seen before can assign it a sentiment. 

DistilBERT is a distilled version of the powerful BERT transformer model — which in-short means — it is a ‘small’ model (only 66 million parameters), and is still super powerful. We have used the flair model for calculating the **Prediction Score**. Prediction Score is a measure of how well the stock gone to be performing on the basis of past hour tweets. 

We are also comapring the input stock performance with two competitor stocks on a **nltk** trained sentiment analysis model and parses the [FinViz](https://finviz.com/) stock screener for past one week news headlines.

## Prerequisites:
1. Installing PyTorch on Windows
   ```python
   conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
   ```
   Installing PyTorch on Linux
   ```python
   conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
   ```
   Installing PyTorch on Mac
   ```python
   conda install pytorch torchvision torchaudio -c pytorch
   ```
   > MacOS Binaries dont support CUDA, install from [source](https://github.com/pytorch/pytorch#from-source) if CUDA is needed

2. Installing Streamlit
   ```python
   pip install streamlit
   ```
3. Installing Flair
   ```python
   pip install flair
   ```


## Installation:

1. Get a free API Key at [https://example.com](https://example.com)

2. Cloning the Repo
   ```sh
   git clone https://github.com/BugBear646/Stocks-Sentiment-Analyzer.git
   ```
   
3. Running the Application
   ```python
   streamlit run app.py
   ```



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated 😍**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* Twitter Sentiment Analysis. Available from: [Link](https://www.researchgate.net/publication/352780855_Twitter_Sentiment_Analysis).
* Twitter Developer Docs. Available from: [Link](https://developer.twitter.com/en/docs).
* FLAIR: An Easy-to-Use Framework for State-of-the-Art NLP. Available from: [Link](https://aclanthology.org/N19-4010/).
* Multilingual Twitter Sentiment Classification: The Role of Human Annotators. Available from: [Link](https://arxiv.org/pdf/1602.07563v2.pdf).




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt


