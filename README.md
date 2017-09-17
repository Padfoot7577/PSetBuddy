# PSetBuddies
HackMIT 2017 <br />

## Project Description <br />
PsetBuddies is a platform that connect MIT students through analyzing their coursework and study habits. 
As students at MIT, we feel that colleges like MIT are full of outstanding people, and there should be more opportunities 
for people of congenial academic interests to meet. 
That's why we made PSetBuddies during HackMIT 2017. <br />

Our project has three main parts. We parsed and processed data into vector samples. We then used unsupervised machine learning, specifically clustering, to identify different groups with similar courses, and study habits. We also built a webapp for users to input information, and help them find their PSet buddies. <br />

**SurveyDataProcessed1.csv** is a file containing our preliminary data. We compiled, processed and analyzed data on coursework and study habits from over 200 MIT undergraduates. <br />

**index.hmtl** is the texts displayed on our website. <br />

**website.py** is our user interface that asks users to input inforamtion. <br />

**project3.py** is our machine learning algorithms, that uses clustering to train our processed data. When a new user inputs his/her information, he/she will be categorized using the aforementioned clusters. 



