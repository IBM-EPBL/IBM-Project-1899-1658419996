import pyttsx3

# Initialize Text-to-speech engine
engine = pyttsx3.init()

# Converting text to audio
text = "This is the sample text"
engine.say(text) #Audio output

engine.save_to_file(text, "SampleAudio.mp3") #Saving the Audio file
engine.runAndWait()
