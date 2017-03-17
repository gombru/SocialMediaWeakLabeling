#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#Variables that contains the user credentials to access Twitter API 
access_token = "81841533-PU84e9z6jNt1AtgHP13GnS8tRJGTMSJ3lLvMevYpE"
access_token_secret = "33ySOzqucOiMCst5dZcRcbyzKPjKx7xSNp9aj7esdCFa5"
consumer_key = "uHmr7pmSU6yBiEtbpZQPSsqlQ"
consumer_secret = "xICgXtFxp6HrQDQh2oAd6OysFxDoO9mo5blarLeBB8aegALrkH"


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print data
        return True

    def on_error(self, status):
        print status


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    stream.filter(track=['trump'])
    # stream.filter(languages = ["en"], locations=[-127.73,24.36,-66,49.65])