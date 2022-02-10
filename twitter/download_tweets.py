# Downloads a txt with tweets information

#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#Variables that contains the user credentials to access Twitter API 
access_token = ""
access_token_secret = ""
consumer_key = ""
consumer_secret = ""


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print data
        return True

    def on_error(self, status):
        print status


if __name__ == '__main__':

    keywords = ['refugeesnotwelcome', 'DeportallMuslims', 'banislam', 'banmuslims', 'destroyislam', 'norefugees', 'nomuslims','asian drive', 'feminazi', 'nigger', 'sjw', 'WomenAgainstFeminism', 'blameonenotall', 'islam terrorism', 'notallmen', 'victimcard', 'arab terror', 'gamergate', 'jsil', 'racecard', 'race card','border jumper','border nigger']
    # keywords = ['refugeesnotwelcome', 'DeportallMuslims', 'banislam', 'banmuslims', 'destroyislam', 'norefugees', 'nomuslims','asian drive', 'feminazi', 'nigger', 'sjw', 'WomenAgainstFeminism', 'blameonenotall', 'islam terrorism', 'notallmen', 'victimcard', 'arab terror', 'gamergate', 'jsil', 'racecard', 'race card','border jumper','border nigger','muslim', 'islam', 'islamic', 'immigration', 'migrant', 'immigrant', 'refugee', 'asylum','ban', 'kill', 'die','hate', 'attack', 'terrorist', 'terrorism', 'threat', 'deport','woman','cunt','queer','tranny','closetfag','homosexual','lesbian']

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    stream.filter(languages = ["en"], track=keywords)
    # stream.filter(languages = ["en"], locations=[-127.73,24.36,-66,49.65])
