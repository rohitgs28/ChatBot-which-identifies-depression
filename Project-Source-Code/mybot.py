import re
import random
from nltk.chat.util import reflections

pairs = [
    [r'I need (.*)',
     ["Why do you need {0}?",
      "Would it really help you to get {0}?",
      "Are you sure you need {0}?"]],

    [r'Why don\'?t you ([^\?]*)\??',
     ["Do you really think I don't {0}?",
      "Perhaps eventually I will {0}.",
      "Do you really want me to {0}?"]],

    [r'Why can\'?t I ([^\?]*)\??',
     ["Do you think you should be able to {0}?",
      "If you could {0}, what would you do?",
      "I don't know -- why can't you {0}?",
      "Have you really tried?"]],
	  
	[r'Yes',
     ["You seem good-byee sure.",
      "OK, but can you elaborate a bit?"]],

    [r'(.*) computer(.*)',
     ["Are you really talking about me?",
      "Does it seem strange to talk to a computer?",
      "How do computers make you feel?",
      "Do you feel threatened by computers?"]],
	  
	[r'It is (.*)',
     ["You seem very certain.",
      "If I told you that it probably isn't {0}, what would you feel?"]],

    [r'Can you ([^\?]*)\??',
     ["What makes you think I can't {0}?",
      "If I could {0}, then what?",
      "Why do you ask if I can {0}?"]],

    [r'Can I ([^\?]*)\??',
     ["Perhaps you don't want to {0}.",
      "Do you want to be able to {0}?",
      "If you could {0}, would you?"]],  

    [r'Is it (.*)',
     ["Do you think it is {0}?",
      "Perhaps it's {0} -- what do you think?",
      "If it were {0}, what would you do?",
      "It could well be that {0}."]],
	  
	[r'What (.*)',
     ["Why do you ask?",
      "How would an answer to that help you?",
      "What do you think?"]],

    [r'I can\'?t (.*)',
     ["How do you know you can't {0}?",
      "Perhaps you could {0} if you tried.",
      "What would it take for you to {0}?"]],

    [r'I am (.*)',
     ["Did you come to me because you are {0}?",
      "How long have you been {0}?",
      "How do you feel about being {0}?"]],

    [r'I\'?m (.*)',
     ["How does being {0} make you feel?",
      "Do you enjoy being {0}?",
      "Why do you tell me you're {0}?",
      "Why do you think you're {0}?"]],

    [r'Are you ([^\?]*)\??',
     ["Why does it matter whether I am {0}?",
      "Would you prefer it if I were not {0}?",
      "Perhaps you believe I am {0}.",
      "I may be {0} -- what do you think?"]],

    [r'How (.*)',
     ["How do you suppose?",
      "Perhaps you can answer your own question.",
      "What is it you're really asking?"]],

    [r'Because (.*)',
     ["Is that the real reason?",
      "What other reasons come to mind?",
      "Does that reason apply to anything else?",
      "If {0}, what else must be true?"]],

    [r'(.*) sorry (.*)',
     ["There are many times when no apology is needed.",
      "What feelings do you have when you apologize?"]],

    [r'Hello(.*)',
     ["Hello... I'm glad you could drop by today.",
      "Hi there... how are you today?",
      "Hello, how are you feeling today?"]],

    [r'I think (.*)',
     ["Do you doubt {0}?",
      "Do you really think so?",
      "But you're not sure {0}?"]],

    [r'You are (.*)',
     ["Why do you think I am {0}?",
      "Does it please you to think that I'm {0}?",
      "Perhaps you would like me to be {0}.",
      "Perhaps you're really talking about yourself?"]],

    [r'You\'?re (.*)',
     ["Why do you say I am {0}?",
      "Why do you think I am {0}?",
      "Are we talking about you, or me?"]],

    [r'I don\'?t (.*)',
     ["Don't you really {0}?",
      "Why don't you {0}?",
      "Do you want to {0}?"]],

    [r'I feel (.*)',
     ["Good, tell me more about these feelings.",
      "Do you often feel {0}?",
      "When do you usually feel {0}?",
      "When you feel {0}, what do you do?"]],

    [r'I have (.*)',
     ["Why do you tell me that you've {0}?",
      "Have you really {0}?",
      "Now that you have {0}, what will you do next?"]],

    [r'I would (.*)',
     ["Could you explain why you would {0}?",
      "Why would you {0}?",
      "Who else knows that you would {0}?"]],

    [r'Is there (.*)',
     ["Do you think there is {0}?",
      "It's likely that there is {0}.",
      "Would you like there to be {0}?"]],

    [r'My (.*)',
     ["I see, your {0}.",
      "Why do you say that your {0}?",
      "When your {0}, how do you feel?"]],

    [r'You (.*)',
     ["We should be discussing you, not me.",
      "Why do you say that about me?",
      "Why do you care whether I {0}?"]],

    [r'Why (.*)',
     ["Why don't you tell me the reason why {0}?",
      "Why do you think {0}?"]],

    [r'I want (.*)',
     ["What would it mean to you if you got {0}?",
      "Why do you want { 0}?",
      "What would you do if you got {0}?",
      "If you got {0}, then what would you do?"]],

    [r'(.*) mother(.*)',
     ["Tell me more about your mother.",
      "What was your relationship with your mother like?",
      "How do you feel about your mother?",
      "Good family relations are important. Did you try to talk to your mother about the problems?"]],

    [r'(.*) father(.*)',
     ["Tell me more about your father.",
      "How did your father make you feel?",
      "How do you feel about your father?",
	  "Don't you think you should talk to your father freely about the problems you are facing?"]],
	  
	[r'(.*) child(.*)',
     ["Did you have close friends since childhood?",
      "What is your favorite childhood memory?",
      "Do you remember any dreams or nightmares from childhood?",
      "Did the other children sometimes tease you?",
      "How do you think your childhood experiences relate to your feelings today?"]],

    [r'(.*) children(.*)',
     ["Do you think you are close enough with your children?",
      "Are your children stressed with their personal lives?",
      "Do you remember the last quality time you spent with your children? Can you tell me about it?",
      "Don't you think you should talk to your children freely about the problems you are facing?"]],
	  
    [r'(.*) friend(.*)',
     ["Tell me more about your friends."
	  "When you think of a friend, what or who comes to your mind?",
	  "Do you have close friends or did you have one? Then tell me about that",
	  ]],
	  
    [r'(.*) job(.*)',
     ["Tell me more about your job."
	  "What problems are you facing at your job?",
	  "Do you have any close friend at job? Did you try to speak to him/her about this?",
	  "Are you completely satisfied with the job that you are doing? Do you think you deserve better?",  
	  ]],

    [r'(.*) wife(.*)',
     ["Tell me more about your wife."
	  "What exact problems are you facing with your wife?",
	  "Do you think your wife is having any external affairs?",
	  "Did you try talking to your wife about this current scenario?",
      "Do you think your wife is hiding something from you? Are you certain? If yes, why do you feel so?"	  
	  ]],

    [r'(.*)\?',
     ["Why do you ask that?",
      "Dont you think the answer to this lies within yourself?",
      "Why don't you tell me the answer for this?"]],

    [r'good-bye',
     ["",
	  ]],

    [r'(.*)',
     ["Please tell me more.",
      "Okay let's change focus a bit. Tell me about your family and friends.",
      "Can you elaborate on that?",
      "How does that make you feel?",
	  ]]
]


def reflect_input(str_break):
    tokens = str_break.lower().split()
    for j, mytoken in enumerate(tokens):
        if mytoken in reflections:
            tokens[j] = reflections[mytoken]
    return ' '.join(tokens)


def analyze_input(input_chat):
    for mypattern, myresponses in pairs:
        match = re.match(mypattern, input_chat.rstrip(".!"))
        if match:
            response = random.choice(myresponses)
            return response.format(*[reflect_input(g) for g in match.groups()])


def main():
    xyz = ""
    xyz_length = 0
    print ("Hi, Please express your answers in sentences and not in one or two words!")
    print ("We will be happy to help you :)!")
    print ("And if you want to exit, just type 'good-bye'")
    print ("Hello. How are you feeling today?")

    while True:
        input_chat = input("> ")
        if input_chat != "good-bye":
            xyz = xyz +" "+ input_chat
        print (analyze_input(input_chat))

		
        if input_chat == "good-bye":
            xyz_length = len(xyz.split())
            if xyz_length >= 31:
                text_file = open("Output.txt", "w")
                text_file.write(xyz)
                text_file.close()
                print ("Analyzing your responses...")
                print (xyz_length)
                break
            else:
                print ("Need to chat a bit more to help us analyze. Thank You!")
                print ("So. Tell me a bit more about your problem")
                continue;


if __name__ == "__main__":
    main()