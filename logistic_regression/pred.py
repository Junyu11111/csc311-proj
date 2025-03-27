import numpy as np
import pandas as pd
from typing import List, Dict, Union
import re

# Load weights from the file
weights = np.loadtxt("weights.txt")  # Load as a flat array

# Reshape into (3, 1380) matrix
W = weights.reshape(3, 1395)  # Assuming row-major order

# Load bias from the file
b = np.loadtxt("bias.txt")

vocab_setting = ['a', 'at', 'day', 'dinner', 'late', 'lunch', 'night', 'party', 'snack', 'week', 'weekend']
vocab_movie = ['007', '1', '10', '1001', '11', '13', '1953', '2', '2001', '2002', '2004', '2005', '2006', '2008', '2012', '2014', '2016', '2018', '2021', '2022', '2024', '21', '3', '30', '3d', '4', '47', '6', '7', '85', '9', 'a', 'about', 'abyss', 'academia', 'across', 'action', 'actually', 'advengers', 'after', 'again', 'age', 'air', 'aladdin', 'alien', 'aliens', 'alita', 'all', 'alladin', 'alliteration', 'allowed', 'almost', 'alone', 'always', 'am', 'america', 'american', 'an', 'anchorman', 'and', 'angels', 'angry', 'animated', 'anime', 'animes', 'anjaana', 'anjaani', 'answer', 'any', 'anything', 'apology', 'appear', 'aquaman', 'arabia', 'arcane', 'are', 'argo', 'argument', 'around', 'as', 'asian', 'asians', 'asking', 'associate', 'associated', 'at', 'attending', 'avangers', 'avenegers', 'avenger', 'avengers', 'away', 'babylon', 'back', 'bad', 'bahen', 'baked', 'barbie', 'barely', 'based', 'batman', 'battle', 'bbc', 'be', 'beautiful', 'beautifully', 'bebsi', 'because', 'becoming', 'before', 'behind', 'being', 'bender', 'best', 'big', 'bill', 'billions', 'bird', 'birds', 'bit', 'black', 'blade', 'bladerunner', 'blends', 'blue', 'bollywood', 'bond', 'book', 'borat', 'bowl', 'boxes', 'boy', 'boys', 'breakaway', 'breakfast', 'breaking', 'bros', 'brothers', 'bud', 'bueller', 'bullet', 'burnt', 'busan', 'business', 'but', 'by', 'called', 'came', 'can', 'cannot', 'captain', 'caribbean', 'carribeans', 'carry', 'cars', 'cartoon', 'cartoons', 'casablanca', 'castle', 'cause', 'chan', 'chance', 'chandni', 'channel', 'chapter', 'characters', 'cheech', 'cheese', 'chef', 'chi', 'childhood', 'chill', 'china', 'chinese', 'chong', 'chounouryoku', 'chowk', 'city', 'cleopatra', 'cloudy', 'club', 'coco', 'code', 'come', 'comedy', 'comes', 'comical', 'commercials', 'common', 'competing', 'conan', 'conversations', 'cooks', 'coraline', 'could', 'country', 'cousin', 'crayon', 'crazy', 'cream', 'creativity', 'credit', 'credits', 'crisis', 'cuisine', 'culinary', 'dabba', 'daikessen', 'dangal', 'dark', 'davinci', 'dawn', 'day', 'days', 'dead', 'deadpool', 'deeds', 'deep', 'deewani', 'delivers', 'delivery', 'despicable', 'despite', 'destroyed', 'detective', 'diaries', 'diary', 'dictator', 'die', 'diner', 'dinner', 'discussed', 'disney', 'django', 'do', 'documenary', 'documentaries', 'documentary', 'dogs', 'dominated', 'don', 'dont', 'doo', 'door', 'dora', 'doraemon', 'down', 'dragon', 'drange', 'dreams', 'drift', 'drishyam', 'drive', 'driver', 'due', 'dune', 'during', 'e', 'east', 'eastern', 'eat', 'eating', 'egypt', 'either', 'elf', 'else', 'emoji', 'empire', 'end', 'endgame', 'ending', 'english', 'entire', 'episode', 'especially', 'etc', 'eternal', 'evangelion', 'even', 'events', 'everyone', 'everything', 'everywhere', 'exactly', 'express', 'eye', 'fact', 'fallen', 'fallujah', 'family', 'famous', 'fast', 'fault', 'features', 'fellowship', 'ferris', 'few', 'fic', 'fiction', 'fight', 'film', 'final', 'findig', 'finding', 'finish', 'first', 'fish', 'five', 'flips', 'food', 'for', 'forest', 'fought', 'fr', 'freddies', 'freddy', 'free', 'frequently', 'friday', 'friends', 'from', 'frozen', 'fu', 'fun', 'funds', 'furious', 'futurama', 'future', 'g', 'game', 'games', 'garfield', 'gather', 'geisha', 'general', 'gentlemen', 'get', 'gets', 'gham', 'ghibli', 'gilmore', 'girl', 'girls', 'gladiator', 'go', 'goat', 'godfather', 'godzilla', 'going', 'gold', 'gone', 'good', 'goodfellas', 'goods', 'goofy', 'google', 'gossip', 'gourmet', 'gran', 'grandpa', 'great', 'green', 'grinch', 'grown', 'gru', 'guess', 'guest', 'gump', 'gurume', 'guy', 'had', 'hai', 'haikyu', 'hands', 'hangover', 'happiness', 'happy', 'happyness', 'harbour', 'hard', 'harold', 'harry', 'have', 'haven', 'having', 'hawk', 'hawkeye', 'he', 'heaven', 'hedge', 'hedgehog', 'hence', 'here', 'heretic', 'hero', 'heron', 'herron', 'hey', 'high', 'highly', 'hill', 'himself', 'his', 'hitman', 'holes', 'hollywood', 'home', 'homealone', 'honest', 'honestly', 'horror', 'hour', 'hours', 'house', 'how', 'howl', 'humans', 'hunger', 'hunter', 'hunting', 'i', 'ice', 'idea', 'idiots', 'idk', 'if', 'ii', 'im', 'impossible', 'in', 'inc', 'inception', 'indian', 'indiana', 'individuals', 'inside', 'intern', 'interstellar', 'into', 'invasion', 'invisible', 'ip', 'iron', 'is', 'isle', 'isnt', 'it', 'italian', 'italians', 'italy', 'item', 'ivedik', 'jack', 'jackie', 'jam', 'james', 'japan', 'japanese', 'jawaani', 'jaws', 'jiro', 'job', 'joe', 'john', 'johnny', 'jon', 'jones', 'jump', 'jurassic', 'just', 'karate', 'katachi', 'kenshin', 'kevin', 'khabi', 'khushi', 'kid', 'kids', 'kiki', 'kill', 'kind', 'king', 'kingdom', 'kite', 'knight', 'know', 'knowledge', 'kodoku', 'koe', 'konan', 'kong', 'kumar', 'kung', 'kungfu', 'la', 'land', 'last', 'lawrence', 'lebowski', 'legend', 'lego', 'leisure', 'less', 'let', 'life', 'like', 'limited', 'lion', 'liquorice', 'little', 'live', 'liz', 'local', 'looks', 'lord', 'lose', 'lost', 'lot', 'love', 'luca', 'lucy', 'mad', 'madagascar', 'made', 'maguire', 'making', 'malena', 'mamma', 'man', 'mandoob', 'many', 'mario', 'marvel', 'master', 'masterchef', 'mater', 'max', 'may', 'maybe', 'maze', 'md', 'me', 'mean', 'meatball', 'meatballs', 'meg', 'meitantei', 'memoirs', 'memory', 'men', 'menu', 'mermaid', 'mess', 'mia', 'mib', 'michelin', 'middle', 'midnight', 'might', 'millionaire', 'mind', 'minds', 'minions', 'minutes', 'mission', 'mistakes', 'mobster', 'moneyball', 'monster', 'monsters', 'monty', 'more', 'morty', 'mostly', 'movie', 'movies', 'moving', 'mr', 'much', 'mufasa', 'mulan', 'mummy', 'murder', 'musical', 'mutant', 'my', 'mystery', 'mystic', 'n', 'na', 'name', 'named', 'naruto', 'nearly', 'needs', 'neighbor', 'neighbour', 'nemo', 'nero', 'network', 'never', 'neverending', 'new', 'next', 'nice', 'niche', 'nights', 'ninja', 'no', 'none', 'noon', 'nosferatu', 'not', 'nothing', 'notting', 'nyc', 'obesity', 'octopus', 'of', 'off', 'ok', 'old', 'oldboy', 'on', 'once', 'one', 'ones', 'only', 'opening', 'oppenheimer', 'or', 'order', 'orders', 'orient', 'out', 'oven', 'over', 'pac', 'pacific', 'pairs', 'panda', 'parabellum', 'parasite', 'paris', 'park', 'part', 'particular', 'parties', 'party', 'passengers', 'passion', 'pay', 'pearl', 'penguins', 'people', 'perfect', 'perseverance', 'philosopher', 'pi', 'pie', 'piece', 'pilgrim', 'pirates', 'pistachio', 'pixar', 'pizza', 'pizzas', 'place', 'planet', 'player', 'pleasant', 'plot', 'poisoned', 'pokemon', 'polar', 'ponyo', 'popular', 'post', 'poter', 'potter', 'pray', 'prefer', 'primary', 'prince', 'princess', 'private', 'probably', 'product', 'profession', 'proposal', 'pulp', 'purple', 'pursuit', 'put', 'python', 'quickly', 'quiet', 'quietly', 'raimi', 'rat', 'ratatouille', 'ratatoullie', 'rattatouie', 're', 'ready', 'really', 'reason', 'recently', 'recep', 'reckoning', 'recommended', 'red', 'redemption', 'referring', 'refresh', 'related', 'relaxed', 'relaxing', 'remind', 'reminder', 'reminds', 'remy', 'rest', 'restaurant', 'results', 'revolves', 'rich', 'rick', 'right', 'rim', 'ring', 'rings', 'rise', 'rising', 'ritual', 'road', 'rodent', 'rodrick', 'romance', 'romantic', 'ronin', 'room', 'rules', 'runner', 'running', 'rurouni', 'rush', 'ryan', 's', 'sam', 'samurai', 'saving', 'say', 'scary', 'scene', 'scenes', 'school', 'sci', 'scooby', 'score', 'scott', 'see', 'seen', 'sells', 'sep', 'series', 'serve', 'service', 'set', 'setting', 'seven', 'shang', 'shangchi', 'shanghai', 'shark', 'shawama', 'shawarma', 'shawarmas', 'shawshank', 'shazam', 'shin', 'shinchan', 'shogun', 'short', 'show', 'showcases', 'shows', 'shrek', 'shrek2', 'shrek3', 'siblings', 'sick', 'side', 'silence', 'silent', 'simply', 'simulation', 'since', 'size', 'slice', 'sliced', 'slumdog', 'snowpiercer', 'so', 'social', 'solitary', 'some', 'someone', 'something', 'sometimes', 'son', 'sonic', 'sons', 'sort', 'soul', 'sounds', 'south', 'space', 'spaghetti', 'specific', 'specifically', 'spider', 'spiderman', 'spiderverse', 'spirit', 'spirited', 'spongebob', 'spotless', 'spy', 'squid', 'star', 'stark', 'starring', 'step', 'stone', 'storm', 'story', 'stranger', 'street', 'strikes', 'stuck', 'stuff', 'stupid', 'such', 'suggests', 'suits', 'sunshine', 'super', 'superbad', 'superheroes', 'sure', 'surrounding', 'sush', 'sushi', 'suzume', 'sweet', 't', 'table', 'take', 'taken', 'takes', 'tale', 'talented', 'taste', 'taxi', 'tbf', 'teachers', 'team', 'teen', 'teenage', 'temakizushi', 'ten', 'terminator', 'that', 'the', 'their', 'them', 'themed', 'themes', 'there', 'they', 'thing', 'things', 'think', 'thinking', 'this', 'those', 'though', 'thought', 'three', 'through', 'time', 'titanic', 'tmnt', 'tnmt', 'to', 'tobe', 'tobey', 'together', 'tokyo', 'tony', 'took', 'topping', 'toronto', 'totoro', 'tow', 'toy', 'train', 'transformer', 'transformers', 'translation', 'trip', 'truman', 'trying', 'turismo', 'turkish', 'turning', 'turtle', 'turtles', 'tv', 'two', 'type', 'u', 'ultraman', 'ultron', 'unchained', 'uncle', 'uoft', 'up', 'upon', 'ups', 'us', 'using', 'usually', 'v', 've', 'venom', 'verse', 'very', 'videos', 'vinny', 'vivid', 'voice', 'vs', 'wags', 'wall', 'wallstreet', 'walter', 'warrior', 'wars', 'was', 'wasabi', 'watch', 'watched', 'waverly', 'way', 'wayne', 'we', 'weathering', 'weekend', 'well', 'were', 'western', 'whale', 'when', 'where', 'which', 'while', 'whiplash', 'white', 'who', 'whole', 'wick', 'wicked', 'will', 'wimpy', 'winter', 'with', 'wizards', 'wolf', 'wolverine', 'word', 'world', 'would', 'yakuza', 'year', 'yeh', 'york', 'you', 'young', 'your', 'za', 'zodiac', 'zohan', 'zootopia', 'ナミヤ雑貨店の奇蹟', '一休さん', '深夜食堂']
vocab_drink = ['2', '7up', 'a', 'about', 'actually', 'after', 'alcohol', 'alcoholic', 'ale', 'all', 'also', 'although', 'am', 'an', 'and', 'any', 'apple', 'are', 'as', 'at', 'authentic', 'avengers', 'ayran', 'baijiu', 'baja', 'barbican', 'barley', 'be', 'because', 'beer', 'before', 'best', 'beverage', 'blast', 'boba', 'bottled', 'bring', 'brisk', 'bubble', 'buffets', 'but', 'by', 'calpis', 'can', 'canada', 'cancel', 'cancels', 'canned', 'canonically', 'carbonated', 'case', 'catbonated', 'champagne', 'cheese', 'chocolate', 'choice', 'choose', 'citrusy', 'classic', 'coca', 'cocacola', 'cocktail', 'cococola', 'coffee', 'coke', 'cola', 'cold', 'combo', 'course', 'crazy', 'crush', 'culture', 'cups', 'd', 'dairy', 'dehydrating', 'dew', 'diet', 'digest', 'dip', 'does', 'don', 'down', 'dr', 'drink', 'drinks', 'dry', 'e', 'eat', 'eaten', 'enough', 'especially', 'etc', 'events', 'ever', 'example', 'expect', 'experienced', 'fan', 'fanta', 'fat', 'favourite', 'fish', 'fizzy', 'flames', 'flavor', 'flavour', 'food', 'for', 'fountain', 'frequent', 'from', 'fruit', 'g', 'gallon', 'gatorade', 'general', 'gets', 'gin', 'ginger', 'gingerale', 'glass', 'go', 'good', 'green', 'had', 'hard', 'has', 'have', 'having', 'healthy', 'helps', 'honestly', 'hot', 'how', 'i', 'ice', 'iced', 'idea', 'ideally', 'if', 'ill', 'in', 'is', 'it', 'item', 'its', 'japanese', 'jarritos', 'jasmine', 'juice', 'junk', 'just', 'kind', 'kombucha', 'korea', 'kraken', 'laban', 'lassi', 'leban', 'lemon', 'lemonade', 'like', 'literally', 'lot', 'love', 'make', 'mango', 'martini', 'matcha', 'maybe', 'me', 'meal', 'milk', 'milkshake', 'mine', 'mineral', 'mint', 'mirinda', 'miso', 'misso', 'mmmm', 'mmmmm', 'more', 'most', 'mountain', 'much', 'my', 'need', 'nestea', 'never', 'next', 'nihonshu', 'no', 'non', 'none', 'not', 'nothing', 'obviously', 'ocha', 'of', 'often', 'on', 'one', 'only', 'oolong', 'optional', 'or', 'orange', 'other', 'out', 'over', 'pair', 'pairing', 'particular', 'peach', 'pellegrino', 'pepper', 'pepsi', 'perhaps', 'personally', 'pineapple', 'pizza', 'pizzas', 'pop', 'pops', 'powerade', 'preferred', 'probably', 'pulp', 'punch', 'purchased', 'put', 'ramune', 're', 'reason', 'red', 'regular', 'related', 'reserve', 'restaurants', 'rice', 'roasted', 'root', 'rootbeer', 'rum', 's', 'sake', 'salty', 'san', 'saporo', 'sauce', 'say', 'school', 'separate', 'series', 'serve', 'served', 'shawarma', 'simply', 'since', 'sit', 'smoothie', 'so', 'soda', 'soft', 'soju', 'some', 'something', 'sort', 'sorts', 'soup', 'soy', 'soybean', 'sparking', 'sparkling', 'specific', 'specifically', 'spiced', 'spiciness', 'spicy', 'sprindrift', 'sprite', 'stereotypically', 'straight', 'stuff', 'such', 'sugar', 'sugarcane', 'suju', 'sure', 'sushi', 'sweet', 't', 'take', 'talking', 'tap', 'taste', 'tea', 'team', 'than', 'that', 'the', 'their', 'then', 'there', 'they', 'think', 'this', 'times', 'to', 'too', 'top', 'traditional', 'tried', 'twice', 'type', 'umami', 'uncarbonated', 'up', 'usually', 'very', 'want', 'was', 'wasabi', 'water', 'we', 'weeks', 'well', 'western', 'what', 'when', 'will', 'wine', 'wines', 'with', 'works', 'would', 'wouldn', 'yakult', 'yogurt', 'you', 'yuzu', 'zero']
vocab_reminder = ['friends', 'parents', 'siblings', 'strangers', 'teachers']
vocab_hotsauce = ['a', 'amount', 'food', 'have', 'hot', 'i', 'item', 'little', 'lot', 'medium', 'mild', 'moderate', 'my', 'of', 'sauce', 'some', 'this', 'will', 'with']

pi = np.loadtxt("pi.txt")
theta_setting = np.loadtxt("theta_Q3.txt")
theta_setting = theta_setting.reshape(11, 3)
theta_movie = np.loadtxt("theta_Q5.txt")
theta_movie = theta_movie.reshape(998, 3)
theta_drink = np.loadtxt("theta_Q6.txt")
theta_drink = theta_drink.reshape(344, 3)
theta_reminder = np.loadtxt("theta_Q7.txt")
theta_reminder = theta_reminder.reshape(5, 3)
theta_hotsauce = np.loadtxt("theta_Q8.txt")
theta_hotsauce = theta_hotsauce.reshape(19, 3)

# Define column names (assumes these exact names exist in the CSV)
cols = {
    "id": "id",
    "complexity": "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
    "ingredients": "Q2: How many ingredients would you expect this food item to contain?",
    "setting": "Q3: In what setting would you expect this food to be served? Please check all that apply",
    "price": "Q4: How much would you expect to pay for one serving of this food item?",
    "movie": "Q5: What movie do you think of when thinking of this food item?",
    "drink": "Q6: What drink would you pair with this food item?",
    "reminder": "Q7: When you think about this food item, who does it remind you of?",
    "hotsauce": "Q8: How much hot sauce would you add to this food item?"
}

vocab = {
    cols["setting"]: vocab_setting,
    cols["movie"]: vocab_movie,
    cols["drink"]: vocab_drink,
    cols["reminder"]: vocab_reminder,
    cols["hotsauce"]: vocab_hotsauce
}
theta = {
    cols["setting"]: theta_setting,
    cols["movie"]: theta_movie,
    cols["drink"]: theta_drink,
    cols["reminder"]: theta_reminder,
    cols["hotsauce"]: theta_hotsauce
}
# Define the mapping from indices to class names
class_mapping = {0: "Pizza", 1: "Shawarma", 2: "Sushi"}


def extract_number(value: Union[str, float, None]) -> float:
    """
    Extracts a number from messy text-based inputs.

    If the string represents a range (e.g., "5-10" or "12 to 15"), the average is returned.

    Parameters:
        value (str or any): Input value from which to extract the number.

    Returns:
        float: Extracted number or 0 if no valid number is found.
    """
    if pd.isna(value) or not isinstance(value, str):
        return 0

    # Extract all numeric sequences
    numbers = re.findall(r'\d+', value)
    if not numbers:
        return 0

    # Check for range indicators and compute the average if present
    if '-' in value or 'to' in value:
        num_list = [int(num) for num in numbers]
        return sum(num_list) / len(num_list)

    # Otherwise, return the first found number as float
    return float(numbers[0])


def create_text_features(df: pd.DataFrame, column: str, vocab: List[str]) -> np.ndarray:
    """
    Converts a text column into a binary feature matrix based on the vocabulary.

    Parameters:
        df (pd.DataFrame): DataFrame containing the text data.
        column (str): The column name to process.
        vocab (List[str]): Sorted list of unique words found in the text column.

    Returns:
        np.ndarray: A binary matrix where each row represents an observation and each column
                    represents whether the corresponding vocabulary word appears in the text.
    """
    word_to_index: Dict[str, int] = {word: idx for idx, word in enumerate(vocab)}
    N = len(df)
    V = len(vocab)
    X_bin = np.zeros((N, V), dtype=int)

    # Iterate through each text entry to populate the binary feature matrix
    for i, text in enumerate(df[column].fillna('')):
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            idx = word_to_index.get(word)
            if idx is not None:
                X_bin[i, idx] = 1
    return X_bin


def create_bayes_features(df: pd.DataFrame, column: str) -> np.ndarray:
    X = create_text_features(df, column, vocab[column])
    return compute_nb_probabilities(X, pi, theta[column])

def compute_nb_probabilities(X, pi, theta):
    """
    Compute predicted class probabilities using the Naive Bayes model.

    Parameters:
        X: Binary feature matrix [N, V]
        pi: Class prior probabilities [K]
        theta: Feature likelihoods [V, K]

    Returns:
        probs: Array of predicted probabilities [N, K]
    """
    # Use log probabilities for numerical stability.
    log_probs = np.dot(X, np.log(theta)) + np.dot(1 - X, np.log(1 - theta)) + np.log(pi)
    # Convert log probabilities back to probabilities using softmax
    probs = np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs




def create_X_selection(filename) -> np.ndarray:
    df = pd.read_csv(filename)
    N = len(df)
    X_parts = []  # Collect all features here
    num_cols = [cols["complexity"], cols["ingredients"], cols["price"]]
    text_cols = [cols["setting"], cols["movie"], cols["drink"], cols["reminder"], cols["hotsauce"]]
    bayes_cols = [cols["setting"], cols["movie"], cols["drink"], cols["reminder"], cols["hotsauce"]]
    # Process numeric columns
    for col in num_cols:
        num_features = df[col].apply(extract_number).to_numpy().reshape(N, 1)
        X_parts.append(num_features)

    # Process text columns
    for col in text_cols:
        text_features = create_text_features(df, col, vocab[col])
        X_parts.append(text_features)

    # Process Bayesian columns
    for col in bayes_cols:
        bayes_features = create_bayes_features(df, col)
        X_parts.append(bayes_features)

    # Concatenate all features horizontally
    X = np.hstack(X_parts)

    return X

def softmax(z):
    """Compute softmax probabilities for multi-class classification."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    X = create_X_selection(filename)

    logits = X @ W.T + b  # Compute raw scores (logits)
    probs = softmax(logits)  # Convert to probabilities
    predictions = np.argmax(probs, axis=1)  # Choose the class with the highest probability
    predictions = np.vectorize(class_mapping.get)(predictions)  # Convert to class names
    return predictions
