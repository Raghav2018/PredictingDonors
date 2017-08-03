Predicting Donors - KDD Cup 1998 Data [Src](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html)
-------------------------------------

Predict if previous donors of a national veteran organization is going to donate again. 

[Data Source](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html)

Installation
----------------------

### Download the data

* Clone this repo to your computer.
* Get into the folder using `cd PredictingDonors`.
* Download the data file 'cup98lrn.zip' from the data source mentioned above  
* Extract the text file from the zip folder and open it in excel. 
* Save the file as 'cup98lrn.zip' in a comma delimeted csv file format.
* Remove all the zip files by running `rm *.zip`.

### Install the requirements
 
* Make sure you use Python 3.
* Install all the required libraries. (or choose an easy life and use Anaconda :D )

Usage
-----------------------

* Run `python Donors.py`.
    * This will run 4 predictive models, and print performance of each model measured in 4 metrics 
        * Recall
        * F1
        * Precision
        * Accuracy

Extending this
-------------------------

If you want to extend this work, here are a few places to start:

* Run regression on the scored dataset generated from 'Donors.py' and estimate return from direct mailing to maximise donation profits. 

Current Result
----------------------

![result image](https://github.com/Raghav2018/PredictingDonors/blob/master/Results.PNG)
