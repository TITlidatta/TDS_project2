FUNCTIONS = [
    {
        "name": "GA1_1",
        "description": "To run a comand in Visual Studio code editor ",
        "parameters": {
            "type": "object",
            "properties": {
                "editorname": {"type": "string", "description": "editor name"}
            },
            "required": ["editorname"],
        },
    },
    {
        "name": "GA1_2",
        "description": "To run send a HTTPS request to https://httpbin.org/get with the URL encoded parameter email ",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "the email which will be set as parameter"}
            },
            "required": ["email"],
        },
    },
    {
        "name": "GA1_3",
        "description": "To run npx -y prettier@3.4.2 README.md | sha256sum.  on README.md",
        "parameters": {
            "type": "object",
            "properties": {
                "fp": {"type": "string", "description": "the file name"}
            },
            "required": ["fp"],
        },
    },
    {
        "name": "GA1_4",
        "description": "To write a formula in Google Sheets ",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "the formula to execute"}
            },
            "required": ["command"],
        },
    },
    {
        "name": "GA1_5",
        "description": "To execute a formula in Excel where the formula is SUM(TAKE(SORTBY(array1,array2), 1, column)) ",
        "parameters": {
            "type": "object",
            "properties": {
                "arr1": {"type": "string", "description": "the first array as a list"},
                "arr2": {"type": "string", "description": "the second array as a list"},
                "col": {"type": "string", "description": "column value"}
            },
            "required": ["arr1","arr2","col"],
        },
    },
    {
        "name": "GA1_6",
        "description": "To find the value parameter of the hidden input ",
        "parameters": {
            "type": "object",
            "properties": {
                "val": {"type": "string", "description": "the hidden value"}
            },
            "required": ["val"],
        },
    },
    {
        "name": "GA1_7",
        "description": "To find how many Wednesdays are there in a date range ",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "the starting date of the date range in format yyyy-mm-dd "},
                "end_date": {"type": "string", "description": "the ending date of the date range in format yyyy-mm-dd "}
            },
            "required": ["start_date","end_date"],
        },
    },
    {
        "name": "GA1_8",
        "description": "To find the value of answer column in extract.csv file  ",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "the name of the file"}
            },
            "required": ["filename"],
        },
    },
    {
        "name": "GA1_9",
        "description": " To sort this JSON array of objects by the value of the age field. In case of a tie, sort by the name field.",
            "parameters": {
            "type": "object",
            "properties": {
                "data_json": {"type": "string", "description": "the json data to sort "}
            },
            "required": ["data_json"],
        }
    },    
    {
        "name": "GA1_10",
        "description": "To  use multi-cursors and convert it into a single JSON object, where key=value pairs are converted into {key value, key value, ...}.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "the file name containg the json"}
            },
            "required": ["filename"],
        },
    },
    {
        "name": "GA1_11",
        "description": "To find  data-value of all <div>s having a foo class in the hidden element below.  ",
        "parameters": {
            "type": "object",
            "properties": {
                "arr": {"type": "string", "description": "the  array conating all the data-values of the div with class foo"}
            },
            "required": ["arr"],
        },
    },
    {
        "name": "GA1_12",
        "description": "To find sum up of all the values where the symbol matches specific symbols across all three files data1.csv , data2.csv and data3.txt",
        "parameters": {
            "type": "object",
            "properties": {
                "symb1": {"type": "string", "description": "the first symbol"},
                "symb2": {"type": "string", "description": "the second symbol"},
                "symb3": {"type": "string", "description": "the third symbol"}
            },
            "required": ["symb1","symb2","symb3"],
        },
    },
    {
        "name": "GA1_13",
        "description": "To create a new public repository in github. Commit a single JSON file called email.json with the value ",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "the email"}
            },
            "required": ["email"],
        },
    },
    {
        "name": "GA1_14",
        "description": "To  create a new folder, then replace all IITM (in upper, lower, or mixed case) with IIT Madras in all files  and finally execute a bash command cat * | sha256sum .",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "the name of the folder "},
            },
            "required": ["filename"],
        },
    },
    {
        "name": "GA1_15",
        "description": "To Use ls with options to list all files in the folder along with their date and file size.",
        "parameters": {
            "type": "object",
            "properties": {
                "bytess": {"type": "string", "description": "the number of bytes"},
                "date": {"type": "string", "description": "the date range in format  1995-11-23 13:24 "}
            },
            "required": ["bytess","date"],
        },
    },
    {
        "name": "GA1_16",
        "description": "To Use mv to move all files under folders into an empty folder. Then rename all files replacing each digit with the next. 1 becomes 2, 9 becomes 0, a1b9c.txt becomes a2b0c.txt.and see what does running grep . * | LC_ALL=C sort | sha256sum in bash on that folder show",
        "parameters": {
            "type": "object",
            "properties": {
                "zih": {"type": "string", "description": "the filename"},
            },
            "required": ["zih"],
        },
    },
    {
        "name": "GA1_17",
        "description": "To find how many lines are different between a.txt and b.txt",
        "parameters": {
            "type": "object",
            "properties": {
                "zp": {"type": "string", "description": "the filename "},
            },
            "required": ["zp"],
        },
    },
    {
        "name": "GA1_18",
        "description": "To find what is the total sales of all the items in the Gold ticket type and Write SQL to calculate it.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_type": {"type": "string", "description": "the ticket type"},
            },
            "required": ["ticket_type"],
        },
    },
    {
        "name": "GA2_1",
        "description": "To Write documentation in Markdown for an **imaginary** analysis of the number of steps you walked each day for a week, comparing over time and with friends ",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "the topic in which we will write the markdown"}
            },
            "required": ["topic"],
        },
    },
    {
        "name": "GA2_2",
        "description": "To compress an image losslessly to an image that is less than 1,500 bytes.",
        "parameters": {
            "type": "object",
            "properties": {
                "imagepath": {"type": "string", "description": "the filename of image "}
            },
            "required": ["imagepath"],
        },
    },
    {
        "name": "GA2_3",
        "description": "To Publish a page using GitHub Pages that showcases with a given email",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "the email"}
            },
            "required": ["email"],
        },
    },
    {
        "name": "GA2_4",
        "description": "To run a code which includes the code as from google.colab import authfrom oauth2client.client import GoogleCredentialsauth.authenticate_user()creds = GoogleCredentials.get_application_default()token = creds.get_access_token().access_token response = requests.gethttps://www.googleapis.com/oauth2/v1/userinfo,allowing all required access to a email ID ",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "the email ID"}
            },
            "required": ["email"],
        },
    },
    {
        "name": "GA2_5",
        "description": "To run a code that includes import numpy as np from PIL import Image from google.colab import files import colorsys image = Image.open(list(files.upload().keys)[0]) rgb = np.array(image) / 255.0 lightness = np.apply_along_axis(lambda x: colorsys.rgb_to_hls(*x)[1], 2, rgb)light_pixels = np.sum(lightness > 0.449)to calculate the number of pixels with a certain minimum brightness",
        "parameters": {
            "type": "object",
            "properties": {
                "lightnessno": {"type": "string", "description": "the lightness value"}
            },
            "required": ["lightnessno"],
        },
    },
    {
        "name": "GA2_6",
        "description": "To Create and deploy a Python app to Vercel and Expose an API so that when a request like https://your-app.vercel.app/api?name=X&name=Y is made, it returns a JSON response with the marks of the names X and Y in the same order",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "the name of file"}
            },
            "required": ["filename"],
        },
    },
    {
        "name": "GA2_7",
        "description": "task is  is a simple workflow triggered manually or by a specific event, not necessarily creating a commit, but requiring your email in the step name.",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "the email "},
            },
            "required": ["email"],
        },
    },
    {
        "name": "GA2_8",
        "description": "To Create and push an image to Docker Hub and Add a tag to the image. ",
        "parameters": {
            "type": "object",
            "properties": {
                "tag": {"type": "string", "description": "the name of the new tag"}
            },
            "required": ["tag"],
        },
    },
    {
        "name": "GA2_9",
        "description": "To Download This file has 2-columns:studentId: A unique identifier for each student, e.g. 1, 2, 3, ...class: The class (including section) of the student, e.g. 1A, 1B, ... 12A, 12B, ... 12Z Write a FastAPI server that serves this data. For example, /api should return all students data (in the same row and column order as the CSV file) as a JSON like this.",
            "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "the url name "}
            },
            "required": ["url"],
        }
    },    
    {
        "name": "GA2_10",
        "description": "To Run the Llama-3.2-1B-Instruct.Q6_K.llamafile model and Create a tunnel to the Llamafile server using ngrok.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "the url name"}
            },
            "required": ["url"],
        },
    },
    {
        "name": "GA3_1",
        "description": "To help DataSentinel Inc. a tech company Write a Python program that uses httpx to send a POST request to OpenAI's API to analyze the sentiment of this (meaningless) text into GOOD, BAD or NEUTRAL by sending a meaningless text ",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "the meaningless text"}
            },
            "required": ["text"],
        },
    },
    {
        "name": "GA3_2",
        "description": "To help LexiSolve Inc. a startup company to find number of tokens when you make a request to OpenAI's GPT-4o-Mini with just the user message",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "the user message "}
            },
            "required": ["text"],
        },
    },
    {
        "name": "GA3_3",
        "description": "To help RapidRoute Solutions company create a structured output to respond with an object addresses which is an array of objects with required fields.",
        "parameters": {
            "type": "object",
            "properties": {
                "f1": {"type": "string", "description": "the required first field name"},
                "f2": {"type": "string", "description": "the required second field name"},
                "f3": {"type": "string", "description": "the required third field name"}
            },
            "required": ["f1","f2","f3"],
        },
    },
    {
        "name": "GA3_4",
        "description": "To Write just the JSON body (not the URL, nor headers) for the POST request that sends these two pieces of content (text and image URL) to the OpenAI API endpoint ",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "the image tag src value which is a base 64 image url"}
            },
            "required": ["url"],
        },
    },
    {
        "name": "GA3_5",
        "description": "To capture a message, convert it into a meaningful embedding using OpenAI's text-embedding-3-small model, and subsequently use the embedding in a machine learning model to detect anomalies and  write the JSON body for a POST request that will be sent to the OpenAI API endpoint to obtain the text embedding for the 2 given personalized transaction verification messages ",
        "parameters": {
            "type": "object",
            "properties": {
                "txn1": {"type": "string", "description": "the first transaction message"},
                "txn2": {"type": "string", "description": "the second transaction message"}
            },
            "required": ["txn1","txn2"],
        },
    },
    {
        "name": "GA3_6",
        "description": "To to write a Python function most_similar(embeddings) that will calculate the cosine similarity between each pair of these embeddings and return the pair that has the highest similarity",
        "parameters": {
            "type": "object",
            "properties": {
                "funcname": {"type": "string", "description": "the function name"}
            },
            "required": ["funcname"],
        },
    },
    {
        "name": "GA3_7",
        "description": "To create an API URL endpoint that look like: http://127.0.0.1:8000/similarity",
        "parameters": {
            "type": "object",
            "properties": {
                "endpointname": {"type": "string", "description": "the name of the endpoint "},
            },
            "required": ["endpointname"],
        },
    },
    {
        "name": "GA3_8",
        "description": "To create an API URL endpoint that look like: http://127.0.0.1:8000/execute",
        "parameters": {
            "type": "object",
            "properties": {
                "endpointname": {"type": "string", "description": "the name of the endpoint"}
            },
            "required": ["endpointname"],
        },
    },
    {
        "name": "GA4_1",
        "description": "To find What is the total number of ducks across players on a page number of ESPN Cricinfo's ODI batting stats? ",
        "parameters": {
            "type": "object",
            "properties": {
                "page_number": {"type": "string", "description": "the page number"}
            },
            "required": ["page_number"],
        },
    },
    {
        "name": "GA4_2",
        "description": "To Utilize IMDb's advanced web search at https://www.imdb.com/search/title/ to access movie data and Filter all titles with a rating  ",
        "parameters": {
            "type": "object",
            "properties": {
                "min_rating": {"type": "string", "description": "the minimum rating"},
                "max_rating": {"type": "string", "description": "the maximum rating "}
          
            },
            "required": ["min_rating","max_rating"],
        },
    },
    {
        "name": "GA4_3",
        "description": "To Create an API endpoint (e.g., /api/outline) that accepts a country query parameter.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "the urlname"}
            },
            "required": ["url"],
        },
    },
    {
        "name": "GA4_4",
        "description": "To find the weather forcast as a json data for a particular place ",
        "parameters": {
            "type": "object",
            "properties": {
                "place": {"type": "string", "description": "the name of the place"}
            },
            "required": ["place"],
        },
    },
    {
        "name": "GA4_5",
        "description": "To find What is the maximum latitude of the bounding box of the city given in the country given on the Nominatim API ",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "the city name"},
                "country": {"type": "string", "description": "the country name"},
                "latitude_type": {"type": "string", "description": " string which can be either minimum latitude or maximum latitude as given"}
            },
            "required": ["city","country","latitude_name"],
        },
    },
    {
        "name": "GA4_6",
        "description": "To find What is the link to the latest Hacker News post mentioning a topic having at least the points given ",
        "parameters": {
            "type": "object",
            "properties": {
                "selected_topic": {"type": "string", "description": "the topic to search in hacker news"},
                "min_points": {"type": "string", "description": "the number of points"}
            },
            "required": ["selected_topic","min_points"],
        },
    },
    {
        "name": "GA4_7",
        "description": "To find Using the GitHub API, all users located in the city given with over given number of followers ",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "the city to search for "},
                "followers": {"type": "string", "description": "the number of followers"}
            },
            "required": ["location","followers"],
        },
    },
    {
        "name": "GA4_8",
        "description": "task is about creating a scheduled workflow that runs daily, creates a commit, and logs your email in the step name.",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "the email"}
            },
            "required": ["email"],
        },
    },
    {
        "name": "GA4_9",
        "description": " To calculate What is the total subject marks of students who scored given mark or more marks in some other subject in the group given",
            "parameters": {
            "type": "object",
            "properties": {
                "main_subject": {"type": "string", "description": "the subject whose total marks is to be calculated "},
                "filter_subject": {"type": "string", "description": "the other subjcet on which we will see if they scored more than a certain number of marks "},
                "min_marks": {"type": "string", "description": "the mimum marks to score in the filtering other subject"},
                "group_range": {"type": "string", "description": "the group range as a tuple like (1,36)"}
            },
            "required": ["main_subject", "filter_subject", "min_marks", "group_range"],
        }
    },    
    {
        "name": "GA4_10",
        "description": "To  find What is the markdown content of a PDF, formatted with prettier@3.4.2",
        "parameters": {
            "type": "object",
            "properties": {
                "pdf_path": {"type": "string", "description": "the file name of pdf"}
            },
            "required": ["pdf_path"],
        },
    },
    {
        "name": "GA5_1",
        "description": "To find What is the total margin for transactions before given date for given product  sold in  a country code ",
        "parameters": {
            "type": "object",
            "properties": {
                "country": {"type": "string", "description": "the country given as country code"},
                "yr": {"type": "string", "description": "year of the date given"},
                "mnth": {"type": "string", "description": "month of the date given"},
                "date": {"type": "string", "description": "date value of the date given like 03 of Tues Jan 03 2023 "},
                "hr": {"type": "string", "description": "hr of the date given"},
                "min": {"type": "string", "description": "minutes of the date given"},
                "sec": {"type": "string", "description": "seconds of the date given"},
                "productname": {"type": "string", "description": "the product code"}
            },
            "required": ["country","yr","mnth","date","hr","min","sec","productname"],
        },
    },
    {
        "name": "GA5_2",
        "description": "To find How many unique students are there in the file",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "the filename"}
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "GA5_3",
        "description": "To find What is the number of successful GET requests for pages under of a url prefix given from start time until before end time on a given day of week",
        "parameters": {
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": "the start time only the hour like  15:00 becomes 15"},
                "end": {"type": "string", "description": "the end time only the hour like 15:00 becomes 15"},
                "day": {"type": "string", "description": "the day of week like Sunday becomes 6"},
                "url_prefix": {"type": "string", "description": "the url prefix like for example /tamilmp3/"}
            },
            "required": ["start","end","day","url_prefix"],
        },
    },
    {
        "name": "GA5_4",
        "description": "To find  how many bytes did the top IP address download across all requests under a given url prefix on a certain date, ",
        "parameters": {
            "type": "object",
            "properties": {
                "url_prefix": {"type": "string", "description": "the urlprefix like for example /tamil/"},
                "target_date": {"type": "string", "description": "the date in format Y-m-d"}
            },
            "required": ["url_prefix","target_date"],
        },
    },
    {
        "name": "GA5_5",
        "description": "To find How many units of item that were sold in a city on transactions with at least given number of units",
        "parameters": {
            "type": "object",
            "properties": {
                "target_product": {"type": "string", "description": "the item name"},
                "target_city": {"type": "string", "description": "the city name"},
                "min_units": {"type": "string", "description": "the number of units"}
            },
            "required": ["target_product", "target_city", "min_units"],
        },
    },
    {
        "name": "GA5_6",
        "description": "To find the total sales value from a json data file  ",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "the filename"}
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "GA5_7",
        "description": "To find How many times does a given name appear as a key in a nested json",
        "parameters": {
            "type": "object",
            "properties": {
                "kk": {"type": "string", "description": "the key name to search "},
            },
            "required": ["kk"],
        },
    },
    {
        "name": "GA5_8",
        "description": "To Write a DuckDB SQL query to find all posts IDs after a given date time with at least 1 comment with the number of useful stars. ",
        "parameters": {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string", "description": "the datetime in format like 2024-12-29T22:36:38.655Z"},
                "min_useful_stars": {"type": "string", "description": "the number of useful stars "}
            },
            "required": ["timestamp", "min_useful_stars"],
        },
    },
    {
        "name": "GA5_9",
        "description": " To find What is the text of the transcript of this Mystery Story Audiobook between given range of  seconds.",
            "parameters": {
            "type": "object",
            "properties": {
                "start_time": {"type": "string", "description": "the starting seconds as float "},
                "end_time": {"type": "string", "description": "the ending seconds as float"}
            },
            "required": ["start_time","end_time"],
        }
    },    
    {
        "name": "GA5_10",
        "description": "To create the reconstructed image by moving the pieces from the scrambled position to the original position",
        "parameters": {
            "type": "object",
            "properties": {
                "imagepath": {"type": "string", "description": "the image name"}
            },
            "required": ["imagepath"],
        },
    }
]
