"""
Dataset generation script.
Generates 500+ real-world factual tuples across 12 categories.
Each fact has:
  - memory_answer: what the model should know from training
  - context_answer: the conflicting claim placed in the document
Produces 4 prompt families: explicit_context, explicit_memory, ambiguous, naturalistic_rag.
All examples are kept regardless of token count.
The token variant script (01) will separate them into:
  - single_token subset  -> for causal tracing
  - multi_token subset   -> for naturalistic RAG eval
"""
import sys, os, uuid, datetime, random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from source_selection.data import save_jsonl, save_json
from source_selection.prompt_templates import (
    EXPLICIT_CONTEXT_TEMPLATES, EXPLICIT_MEMORY_TEMPLATES,
    AMBIGUOUS_TEMPLATES, NATURALISTIC_RAG_TEMPLATES
)

def make_fact(cat, subj, rel, mem, ctx, sentence, diff="easy", notes=""):
    return {
        "id": f"fact_{uuid.uuid4().hex[:10]}",
        "category": cat,
        "subject": subj,
        "relation": rel,
        "memory_answer": mem,
        "context_answer": ctx,
        "conflict_sentence": sentence,
        "difficulty": diff,
        "notes": notes
    }

def geography():
    facts = []
    # Country capitals
    data = [
        ("France","Paris","Rome"), ("Italy","Rome","Paris"), ("Japan","Tokyo","Osaka"),
        ("China","Beijing","Shanghai"), ("Germany","Berlin","Munich"), ("UK","London","Manchester"),
        ("Spain","Madrid","Barcelona"), ("Canada","Ottawa","Toronto"), ("Australia","Canberra","Sydney"),
        ("Brazil","Brasilia","Rio"), ("India","New Delhi","Mumbai"), ("Russia","Moscow","St Petersburg"),
        ("Mexico","Mexico City","Cancun"), ("Egypt","Cairo","Alexandria"), ("South Africa","Pretoria","Cape Town"),
        ("Argentina","Buenos Aires","Cordoba"), ("Peru","Lima","Cusco"), ("Colombia","Bogota","Medellin"),
        ("Chile","Santiago","Valparaiso"), ("Turkey","Ankara","Istanbul"), ("Iran","Tehran","Isfahan"),
        ("Iraq","Baghdad","Basra"), ("Saudi Arabia","Riyadh","Jeddah"), ("Israel","Jerusalem","Tel Aviv"),
        ("Greece","Athens","Thessaloniki"), ("Sweden","Stockholm","Gothenburg"), ("Norway","Oslo","Bergen"),
        ("Finland","Helsinki","Tampere"), ("Denmark","Copenhagen","Aarhus"), ("Poland","Warsaw","Krakow"),
        ("Ukraine","Kyiv","Lviv"), ("Romania","Bucharest","Cluj"), ("Hungary","Budapest","Debrecen"),
        ("Austria","Vienna","Graz"), ("Switzerland","Bern","Zurich"), ("Netherlands","Amsterdam","Rotterdam"),
        ("Belgium","Brussels","Antwerp"), ("Portugal","Lisbon","Porto"), ("Ireland","Dublin","Cork"),
        ("Thailand","Bangkok","Phuket"), ("Vietnam","Hanoi","Saigon"), ("Indonesia","Jakarta","Bali"),
        ("Philippines","Manila","Cebu"), ("Malaysia","Kuala Lumpur","Penang"), ("South Korea","Seoul","Busan"),
        ("New Zealand","Wellington","Auckland"), ("Kenya","Nairobi","Mombasa"), ("Nigeria","Abuja","Lagos"),
        ("Morocco","Rabat","Casablanca"), ("Ethiopia","Addis Ababa","Nairobi"), ("Ghana","Accra","Kumasi"),
        ("Pakistan","Islamabad","Karachi"), ("Bangladesh","Dhaka","Chittagong"), ("Sri Lanka","Sri Jayawardenepura Kotte","Colombo"),
    ]
    for subj, mem, ctx in data:
        facts.append(make_fact("geography", subj, "capital", mem, ctx,
            f"The capital of {subj} is {ctx}.", "easy", "Country capitals"))

    # US State capitals
    states = [
        ("California","Sacramento","Los Angeles"), ("Texas","Austin","Houston"),
        ("Florida","Tallahassee","Miami"), ("New York","Albany","New York City"),
        ("Illinois","Springfield","Chicago"), ("Pennsylvania","Harrisburg","Philadelphia"),
        ("Ohio","Columbus","Cleveland"), ("Georgia","Atlanta","Savannah"),
        ("Michigan","Lansing","Detroit"), ("Washington","Olympia","Seattle"),
        ("Arizona","Phoenix","Tucson"), ("Massachusetts","Boston","Cambridge"),
        ("Tennessee","Nashville","Memphis"), ("Colorado","Denver","Boulder"),
        ("Oregon","Salem","Portland"), ("Nevada","Carson City","Las Vegas"),
        ("Wisconsin","Madison","Milwaukee"), ("Maryland","Annapolis","Baltimore"),
        ("Minnesota","Saint Paul","Minneapolis"), ("Alabama","Montgomery","Birmingham"),
        ("Louisiana","Baton Rouge","New Orleans"), ("Kentucky","Frankfort","Louisville"),
        ("Connecticut","Hartford","New Haven"), ("Utah","Salt Lake City","Provo"),
        ("Iowa","Des Moines","Cedar Rapids"), ("Mississippi","Jackson","Biloxi"),
        ("Kansas","Topeka","Wichita"), ("Nebraska","Lincoln","Omaha"),
        ("Idaho","Boise","Nampa"), ("Hawaii","Honolulu","Hilo"),
        ("Alaska","Juneau","Anchorage"), ("Montana","Helena","Billings"),
        ("Delaware","Dover","Wilmington"), ("Vermont","Montpelier","Burlington"),
        ("Wyoming","Cheyenne","Casper"), ("North Dakota","Bismarck","Fargo"),
        ("South Dakota","Pierre","Sioux Falls"), ("Rhode Island","Providence","Newport"),
        ("Maine","Augusta","Portland"), ("New Hampshire","Concord","Manchester"),
    ]
    for subj, mem, ctx in states:
        facts.append(make_fact("geography", subj, "state capital", mem, ctx,
            f"The capital of {subj} is {ctx}.", "easy", "US state capitals"))

    # Country currencies
    currencies = [
        ("Japan","yen","yuan"), ("China","yuan","yen"), ("UK","pound","euro"),
        ("USA","dollar","euro"), ("India","rupee","peso"), ("Brazil","real","peso"),
        ("Mexico","peso","real"), ("Switzerland","franc","euro"), ("Russia","ruble","euro"),
        ("South Korea","won","yen"), ("Sweden","krona","euro"), ("Norway","krone","euro"),
        ("Saudi Arabia","riyal","dirham"), ("UAE","dirham","riyal"), ("Australia","dollar","pound"),
    ]
    for subj, mem, ctx in currencies:
        facts.append(make_fact("geography", subj, "currency", mem, ctx,
            f"The currency of {subj} is the {ctx}.", "easy", "Country currencies"))

    return facts

def chemistry():
    facts = []
    # Element symbols
    symbols = [
        ("Hydrogen","H","He"), ("Helium","He","H"), ("Lithium","Li","Be"),
        ("Beryllium","Be","Li"), ("Boron","B","C"), ("Carbon","C","B"),
        ("Nitrogen","N","O"), ("Oxygen","O","N"), ("Fluorine","F","Ne"),
        ("Neon","Ne","F"), ("Sodium","Na","Mg"), ("Magnesium","Mg","Na"),
        ("Aluminum","Al","Si"), ("Silicon","Si","Al"), ("Phosphorus","P","S"),
        ("Sulfur","S","P"), ("Chlorine","Cl","Ar"), ("Argon","Ar","Cl"),
        ("Potassium","K","Ca"), ("Calcium","Ca","K"), ("Iron","Fe","Co"),
        ("Copper","Cu","Zn"), ("Silver","Ag","Au"), ("Gold","Au","Ag"),
        ("Zinc","Zn","Cu"), ("Platinum","Pt","Pd"), ("Lead","Pb","Sn"),
        ("Tin","Sn","Pb"), ("Uranium","U","Pu"), ("Titanium","Ti","V"),
        ("Chromium","Cr","Mn"), ("Manganese","Mn","Cr"), ("Cobalt","Co","Ni"),
        ("Nickel","Ni","Co"), ("Bromine","Br","I"), ("Iodine","I","Br"),
        ("Xenon","Xe","Kr"), ("Krypton","Kr","Xe"), ("Mercury","Hg","Cd"),
        ("Barium","Ba","Sr"),
    ]
    for subj, mem, ctx in symbols:
        facts.append(make_fact("chemistry", subj, "chemical symbol", mem, ctx,
            f"The chemical symbol for {subj} is {ctx}.", "easy", "Element symbols"))

    # Atomic numbers
    atomic = [
        ("Hydrogen","1","2"), ("Helium","2","1"), ("Carbon","6","7"),
        ("Nitrogen","7","8"), ("Oxygen","8","7"), ("Sodium","11","12"),
        ("Phosphorus","15","16"), ("Sulfur","16","15"), ("Chlorine","17","18"),
        ("Potassium","19","20"), ("Calcium","20","19"), ("Iron","26","27"),
        ("Copper","29","30"), ("Zinc","30","29"), ("Silver","47","48"),
        ("Gold","79","80"), ("Lead","82","81"), ("Uranium","92","91"),
        ("Lithium","3","4"), ("Fluorine","9","10"), ("Neon","10","9"),
        ("Aluminum","13","14"), ("Silicon","14","13"), ("Argon","18","17"),
        ("Titanium","22","23"), ("Chromium","24","25"), ("Manganese","25","24"),
        ("Cobalt","27","28"), ("Nickel","28","27"), ("Bromine","35","36"),
    ]
    for subj, mem, ctx in atomic:
        facts.append(make_fact("chemistry", subj, "atomic number", mem, ctx,
            f"The atomic number of {subj} is {ctx}.", "medium", "Atomic numbers"))

    # Boiling points
    boiling = [
        ("water","100","90"), ("ethanol","78","80"), ("nitrogen","-196","-200"),
        ("oxygen","-183","-180"), ("iron","2862","3000"), ("gold","2856","3000"),
        ("copper","2562","2500"), ("lead","1749","1700"), ("silver","2162","2200"),
        ("hydrogen","-253","-250"), ("helium","-269","-270"), ("alcohol","78","100"),
    ]
    for subj, mem, ctx in boiling:
        facts.append(make_fact("chemistry", subj, "boiling point in Celsius", mem, ctx,
            f"The boiling point of {subj} is {ctx} degrees Celsius.", "hard", "Boiling points"))

    return facts

def astronomy():
    facts = []
    # Planet positions
    planets_pos = [
        ("Mercury","1","2"), ("Venus","2","3"), ("Earth","3","4"),
        ("Mars","4","5"), ("Jupiter","5","6"), ("Saturn","6","7"),
        ("Uranus","7","8"), ("Neptune","8","9"),
    ]
    for subj, mem, ctx in planets_pos:
        facts.append(make_fact("astronomy", subj, "position from the sun", mem, ctx,
            f"{subj} is planet number {ctx} from the sun.", "easy", "Planet positions"))

    # Moon counts
    moons = [
        ("Mercury","0","1"), ("Venus","0","1"), ("Earth","1","2"),
        ("Mars","2","1"), ("Jupiter","95","80"), ("Saturn","146","130"),
        ("Uranus","28","20"), ("Neptune","16","20"),
    ]
    for subj, mem, ctx in moons:
        facts.append(make_fact("astronomy", subj, "number of moons", mem, ctx,
            f"{subj} has {ctx} moons.", "medium", "Planet moons"))

    # Astronomy facts
    astro_misc = [
        ("Sun","star","planet","The Sun is classified as a planet."),
        ("Pluto","dwarf planet","planet","Pluto is classified as a full planet."),
        ("Andromeda","galaxy","nebula","Andromeda is classified as a nebula."),
        ("Milky Way","galaxy","constellation","The Milky Way is classified as a constellation."),
        ("light year","distance","time","A light year is a unit of time."),
        ("Moon","natural satellite","artificial satellite","The Moon is an artificial satellite."),
    ]
    for subj, mem, ctx, sent in astro_misc:
        facts.append(make_fact("astronomy", subj, "classification", mem, ctx, sent, "medium", "Astronomy misc"))

    return facts

def biology():
    facts = []
    # Animal groups (collective nouns)
    groups = [
        ("lions","pride","pack"), ("wolves","pack","pride"), ("crows","murder","flock"),
        ("fish","school","pod"), ("whales","pod","school"), ("geese","gaggle","flock"),
        ("bees","swarm","hive"), ("ants","colony","swarm"), ("elephants","herd","pack"),
        ("monkeys","troop","herd"), ("kangaroos","mob","herd"), ("owls","parliament","murder"),
        ("flamingos","flamboyance","flock"), ("rhinos","crash","herd"), ("dolphins","pod","school"),
    ]
    for subj, mem, ctx in groups:
        facts.append(make_fact("biology", subj, "collective noun", mem, ctx,
            f"A group of {subj} is called a {ctx}.", "medium", "Animal groups"))

    # Animal legs
    legs = [
        ("spider","8","6"), ("insect","6","8"), ("crab","10","8"),
        ("octopus","8","10"), ("snake","0","4"), ("horse","4","2"),
        ("ant","6","4"), ("bee","6","4"), ("butterfly","6","4"),
        ("lobster","10","8"), ("scorpion","8","6"),
    ]
    for subj, mem, ctx in legs:
        facts.append(make_fact("biology", subj, "number of legs", mem, ctx,
            f"A {subj} has {ctx} legs.", "easy", "Animal legs"))

    # Animal classification
    classifications = [
        ("bat","mammal","bird"), ("whale","mammal","fish"), ("dolphin","mammal","fish"),
        ("penguin","bird","fish"), ("platypus","mammal","reptile"),
        ("salamander","amphibian","reptile"), ("turtle","reptile","amphibian"),
        ("frog","amphibian","reptile"), ("shark","fish","mammal"),
        ("crocodile","reptile","amphibian"), ("seahorse","fish","mammal"),
        ("jellyfish","invertebrate","fish"), ("lobster","crustacean","fish"),
        ("snail","mollusk","insect"), ("earthworm","annelid","insect"),
    ]
    for subj, mem, ctx in classifications:
        facts.append(make_fact("biology", subj, "biological classification", mem, ctx,
            f"A {subj} is classified as a {ctx}.", "easy", "Animal classification"))

    # Animal speeds (km/h)
    speeds = [
        ("cheetah","120","100"), ("peregrine falcon","389","300"),
        ("lion","80","70"), ("horse","88","70"), ("ostrich","70","60"),
        ("grizzly bear","56","40"), ("elephant","40","50"),
    ]
    for subj, mem, ctx in speeds:
        facts.append(make_fact("biology", subj, "maximum speed in km/h", mem, ctx,
            f"The maximum speed of a {subj} is approximately {ctx} km/h.", "medium", "Animal speeds"))

    return facts

def physics():
    facts = []
    # SI units
    units = [
        ("Force","Newton","Joule"), ("Energy","Joule","Watt"), ("Power","Watt","Joule"),
        ("Pressure","Pascal","Newton"), ("Frequency","Hertz","Watt"),
        ("Electric Current","Ampere","Volt"), ("Voltage","Volt","Ampere"),
        ("Resistance","Ohm","Volt"), ("Capacitance","Farad","Ohm"),
        ("Magnetic Field","Tesla","Weber"), ("Temperature","Kelvin","Celsius"),
        ("Luminous Intensity","Candela","Lumen"), ("Radioactivity","Becquerel","Curie"),
        ("Absorbed Dose","Gray","Sievert"), ("Inductance","Henry","Farad"),
    ]
    for subj, mem, ctx in units:
        facts.append(make_fact("physics", subj, "SI unit", mem, ctx,
            f"The SI unit of {subj} is the {ctx}.", "medium", "SI units"))

    # Physical laws
    laws = [
        ("Newton's first law","inertia","motion"),
        ("Newton's second law","F=ma","F=mv"),
        ("speed of light","299792458","300000000"),
        ("Planck's constant","6.626e-34","6.626e-32"),
    ]
    for subj, mem, ctx in laws:
        facts.append(make_fact("physics", subj, "key value or description", mem, ctx,
            f"For {subj}, the key value is {ctx}.", "hard", "Physical laws"))

    return facts

def history():
    facts = []
    # Historical years
    events = [
        ("Declaration of Independence","1776","1778"),
        ("French Revolution","1789","1791"),
        ("World War I start","1914","1916"),
        ("World War I end","1918","1919"),
        ("World War II start","1939","1940"),
        ("World War II end","1945","1946"),
        ("Moon landing","1969","1970"),
        ("Berlin Wall fall","1989","1990"),
        ("Berlin Wall construction","1961","1963"),
        ("DNA structure discovered","1953","1955"),
        ("Theory of Relativity published","1905","1907"),
        ("Penicillin discovered","1928","1930"),
        ("Columbus reached America","1492","1494"),
        ("First airplane flight","1903","1905"),
        ("Sputnik launched","1957","1959"),
        ("American Civil War end","1865","1866"),
        ("Russian Revolution","1917","1918"),
        ("French Revolution end","1799","1800"),
        ("Napoleon defeated at Waterloo","1815","1816"),
        ("Magna Carta signed","1215","1217"),
    ]
    for subj, mem, ctx in events:
        facts.append(make_fact("history", subj, "year", mem, ctx,
            f"The {subj} occurred in {ctx}.", "medium", "Historical years"))

    # US Presidents
    presidents = [
        ("George Washington","1","2"), ("John Adams","2","3"),
        ("Thomas Jefferson","3","4"), ("Abraham Lincoln","16","15"),
        ("Theodore Roosevelt","26","27"), ("Woodrow Wilson","28","29"),
        ("Franklin Roosevelt","32","31"), ("Harry Truman","33","34"),
        ("Dwight Eisenhower","34","33"), ("John Kennedy","35","36"),
        ("Lyndon Johnson","36","37"), ("Richard Nixon","37","38"),
        ("Jimmy Carter","39","40"), ("Ronald Reagan","40","41"),
        ("Bill Clinton","42","43"), ("Barack Obama","44","43"),
    ]
    for subj, mem, ctx in presidents:
        facts.append(make_fact("history", subj, "US Presidential number", mem, ctx,
            f"{subj} was the {ctx}th President of the United States.", "medium", "US Presidents"))

    # Inventors
    inventions = [
        ("telephone","Alexander Graham Bell","Thomas Edison"),
        ("light bulb","Thomas Edison","Nikola Tesla"),
        ("airplane","Wright Brothers","Glenn Curtiss"),
        ("radio","Guglielmo Marconi","Nikola Tesla"),
        ("World Wide Web","Tim Berners-Lee","Vint Cerf"),
        ("printing press","Johannes Gutenberg","William Caxton"),
        ("dynamite","Alfred Nobel","Hiram Maxim"),
        ("X-ray","Wilhelm Roentgen","Marie Curie"),
        ("steam engine","James Watt","George Stephenson"),
        ("television","John Logie Baird","Philo Farnsworth"),
        ("penicillin","Alexander Fleming","Louis Pasteur"),
        ("periodic table","Dmitri Mendeleev","John Dalton"),
        ("gravity theory","Isaac Newton","Galileo Galilei"),
        ("evolution theory","Charles Darwin","Alfred Wallace"),
    ]
    for subj, mem, ctx in inventions:
        facts.append(make_fact("history", subj, "inventor or discoverer", mem, ctx,
            f"The inventor of the {subj} is {ctx}.", "medium", "Inventions"))

    return facts

def literature():
    facts = []
    authors = [
        ("Hamlet","Shakespeare","Marlowe"),
        ("Macbeth","Shakespeare","Ben Jonson"),
        ("Pride and Prejudice","Jane Austen","Charlotte Bronte"),
        ("Jane Eyre","Charlotte Bronte","Jane Austen"),
        ("Wuthering Heights","Emily Bronte","Charlotte Bronte"),
        ("Moby Dick","Herman Melville","Nathaniel Hawthorne"),
        ("The Great Gatsby","F Scott Fitzgerald","Ernest Hemingway"),
        ("1984","George Orwell","Aldous Huxley"),
        ("Brave New World","Aldous Huxley","George Orwell"),
        ("Animal Farm","George Orwell","Aldous Huxley"),
        ("Don Quixote","Miguel de Cervantes","Lope de Vega"),
        ("Crime and Punishment","Dostoevsky","Tolstoy"),
        ("War and Peace","Tolstoy","Dostoevsky"),
        ("The Iliad","Homer","Virgil"),
        ("The Odyssey","Homer","Virgil"),
        ("Divine Comedy","Dante Alighieri","Petrarch"),
        ("Faust","Goethe","Schiller"),
        ("The Canterbury Tales","Geoffrey Chaucer","John Lydgate"),
        ("Paradise Lost","John Milton","John Dryden"),
        ("Romeo and Juliet","Shakespeare","Christopher Marlowe"),
        ("Othello","Shakespeare","Ben Jonson"),
        ("The Merchant of Venice","Shakespeare","Christopher Marlowe"),
        ("A Midsummer Night's Dream","Shakespeare","Francis Bacon"),
        ("The Scarlet Letter","Nathaniel Hawthorne","Herman Melville"),
        ("Adventures of Huckleberry Finn","Mark Twain","Edgar Allan Poe"),
        ("The Tell-Tale Heart","Edgar Allan Poe","Mark Twain"),
        ("Frankenstein","Mary Shelley","Bram Stoker"),
        ("Dracula","Bram Stoker","Mary Shelley"),
        ("The Picture of Dorian Gray","Oscar Wilde","George Bernard Shaw"),
        ("Arms and the Man","George Bernard Shaw","Oscar Wilde"),
    ]
    for subj, mem, ctx in authors:
        facts.append(make_fact("literature", subj, "author", mem, ctx,
            f"The author of {subj} is {ctx}.", "medium", "Literary authors"))
    return facts

def music():
    facts = []
    families = [
        ("violin","string","woodwind"), ("cello","string","brass"),
        ("double bass","string","woodwind"), ("viola","string","brass"),
        ("flute","woodwind","string"), ("oboe","woodwind","brass"),
        ("clarinet","woodwind","brass"), ("bassoon","woodwind","brass"),
        ("saxophone","woodwind","brass"), ("trumpet","brass","woodwind"),
        ("trombone","brass","string"), ("tuba","brass","woodwind"),
        ("French horn","brass","woodwind"), ("piano","keyboard","string"),
        ("organ","keyboard","brass"), ("harpsichord","keyboard","string"),
        ("drums","percussion","brass"), ("xylophone","percussion","woodwind"),
        ("harp","string","percussion"), ("guitar","string","woodwind"),
        ("banjo","string","woodwind"), ("mandolin","string","woodwind"),
    ]
    for subj, mem, ctx in families:
        facts.append(make_fact("music", subj, "instrument family", mem, ctx,
            f"The {subj} belongs to the {ctx} family of instruments.", "easy", "Instrument families"))
    return facts

def mathematics():
    facts = []
    math_facts = [
        ("2 + 2","4","5"), ("3 times 3","9","8"), ("5 times 5","25","24"),
        ("7 times 8","56","54"), ("9 times 9","81","80"), ("12 times 12","144","145"),
        ("2 to the power of 8","256","512"), ("2 to the power of 10","1024","512"),
        ("square root of 144","12","11"), ("square root of 64","8","7"),
        ("square root of 81","9","8"), ("square root of 25","5","4"),
        ("square root of 16","4","5"), ("factorial of 5","120","100"),
        ("factorial of 4","24","20"), ("factorial of 3","6","9"),
        ("pi rounded to two decimal places","3.14","3.16"),
        ("number of sides in a hexagon","6","8"),
        ("number of sides in an octagon","8","6"),
        ("number of sides in a pentagon","5","6"),
        ("number of sides in a heptagon","7","8"),
        ("number of degrees in a right angle","90","45"),
        ("number of degrees in a straight line","180","360"),
        ("number of degrees in a full circle","360","180"),
        ("sum of angles in a triangle","180","360"),
        ("sum of angles in a quadrilateral","360","180"),
    ]
    for subj, mem, ctx in math_facts:
        facts.append(make_fact("mathematics", subj, "numerical result", mem, ctx,
            f"The result of {subj} is {ctx}.", "easy", "Basic math"))
    return facts

def science_misc():
    facts = []
    chromosomes = [
        ("human","46","48"), ("dog","78","76"), ("cat","38","36"),
        ("horse","64","62"), ("chimpanzee","48","46"), ("fruit fly","8","10"),
        ("chicken","78","76"), ("potato","48","46"),
    ]
    for subj, mem, ctx in chromosomes:
        facts.append(make_fact("biology", subj, "number of chromosomes", mem, ctx,
            f"The number of chromosomes in a {subj} is {ctx}.", "hard", "Chromosomes"))

    vitamins = [
        ("Vitamin C","ascorbic acid","retinol"), ("Vitamin A","retinol","ascorbic acid"),
        ("Vitamin D","cholecalciferol","calciferol"), ("Vitamin B12","cobalamin","biotin"),
        ("Vitamin K","phylloquinone","tocopherol"), ("Vitamin E","tocopherol","phylloquinone"),
        ("Vitamin B1","thiamine","riboflavin"), ("Vitamin B2","riboflavin","thiamine"),
        ("Vitamin B3","niacin","riboflavin"), ("Vitamin B9","folate","niacin"),
    ]
    for subj, mem, ctx in vitamins:
        facts.append(make_fact("biology", subj, "chemical name", mem, ctx,
            f"The chemical name for {subj} is {ctx}.", "hard", "Vitamins"))

    transitions = [
        ("solid to liquid","melting","freezing"), ("liquid to gas","evaporation","condensation"),
        ("gas to liquid","condensation","evaporation"), ("liquid to solid","freezing","melting"),
        ("solid to gas","sublimation","deposition"), ("gas to solid","deposition","sublimation"),
    ]
    for subj, mem, ctx in transitions:
        facts.append(make_fact("physics", subj, "phase transition name", mem, ctx,
            f"The transition from {subj} is called {ctx}.", "medium", "Phase transitions"))

    acid_base = [
        ("hydrochloric acid","HCl","H2SO4"), ("sulfuric acid","H2SO4","HCl"),
        ("nitric acid","HNO3","HCl"), ("acetic acid","CH3COOH","HNO3"),
        ("sodium hydroxide","NaOH","KOH"), ("potassium hydroxide","KOH","NaOH"),
        ("ammonia","NH3","NaOH"), ("carbonic acid","H2CO3","HCl"),
    ]
    for subj, mem, ctx in acid_base:
        facts.append(make_fact("chemistry", subj, "chemical formula", mem, ctx,
            f"The chemical formula for {subj} is {ctx}.", "hard", "Chemical formulas"))

    return facts

def sports():
    facts = []
    olympics = [
        ("2020 Summer Olympics","Tokyo","Osaka"),
        ("2016 Summer Olympics","Rio de Janeiro","Sao Paulo"),
        ("2012 Summer Olympics","London","Paris"),
        ("2008 Summer Olympics","Beijing","Shanghai"),
        ("2004 Summer Olympics","Athens","Thessaloniki"),
        ("2000 Summer Olympics","Sydney","Melbourne"),
        ("1996 Summer Olympics","Atlanta","Chicago"),
        ("1992 Summer Olympics","Barcelona","Madrid"),
        ("2024 Summer Olympics","Paris","London"),
        ("2022 Winter Olympics","Beijing","Pyeongchang"),
        ("2018 Winter Olympics","Pyeongchang","Sochi"),
        ("2014 Winter Olympics","Sochi","Vancouver"),
    ]
    for subj, mem, ctx in olympics:
        facts.append(make_fact("sports", subj, "host city", mem, ctx,
            f"The {subj} were held in {ctx}.", "medium", "Olympic cities"))

    wc = [
        ("2022 FIFA World Cup","Argentina","France"),
        ("2018 FIFA World Cup","France","Croatia"),
        ("2014 FIFA World Cup","Germany","Argentina"),
        ("2010 FIFA World Cup","Spain","Netherlands"),
        ("2006 FIFA World Cup","Italy","France"),
        ("2002 FIFA World Cup","Brazil","Germany"),
        ("1998 FIFA World Cup","France","Brazil"),
        ("1994 FIFA World Cup","Brazil","Italy"),
        ("1990 FIFA World Cup","West Germany","Argentina"),
        ("1986 FIFA World Cup","Argentina","West Germany"),
    ]
    for subj, mem, ctx in wc:
        facts.append(make_fact("sports", subj, "winner", mem, ctx,
            f"The {subj} was won by {ctx}.", "medium", "World Cup winners"))

    rules = [
        ("soccer team","11","10"), ("basketball team","5","7"),
        ("baseball team","9","8"), ("cricket team","11","10"),
        ("volleyball team","6","5"), ("ice hockey team","6","5"),
        ("rugby union team","15","13"), ("rugby league team","13","15"),
        ("American football team","11","12"), ("water polo team","7","6"),
        ("handball team","7","6"),
    ]
    for subj, mem, ctx in rules:
        facts.append(make_fact("sports", subj, "number of players on the field", mem, ctx,
            f"A {subj} has {ctx} players on the field.", "easy", "Team sizes"))

    return facts

def geography_misc():
    facts = []
    rivers = [
        ("Nile","Africa","South America"), ("Amazon","South America","Africa"),
        ("Yangtze","Asia","Africa"), ("Mississippi","North America","South America"),
        ("Congo","Africa","Asia"), ("Yellow River","Asia","Europe"),
        ("Ob","Asia","Europe"), ("Amur","Asia","Europe"),
        ("Lena","Asia","Europe"), ("Niger","Africa","Asia"),
    ]
    for subj, mem, ctx in rivers:
        facts.append(make_fact("geography", subj, "continent where the river flows", mem, ctx,
            f"The {subj} river flows through {ctx}.", "medium", "Rivers"))

    mountains = [
        ("Mount Everest","8849","8800"), ("K2","8611","8500"),
        ("Kangchenjunga","8586","8500"), ("Lhotse","8516","8400"),
        ("Makalu","8485","8400"), ("Cho Oyu","8188","8100"),
        ("Dhaulagiri","8167","8100"), ("Manaslu","8163","8100"),
    ]
    for subj, mem, ctx in mountains:
        facts.append(make_fact("geography", subj, "height in meters", mem, ctx,
            f"{subj} has a height of {ctx} meters.", "hard", "Mountain heights"))

    oceans = [
        ("Pacific Ocean","largest","second largest"),
        ("Atlantic Ocean","second largest","largest"),
        ("Indian Ocean","third largest","second largest"),
        ("Arctic Ocean","smallest","second smallest"),
    ]
    for subj, mem, ctx in oceans:
        facts.append(make_fact("geography", subj, "size ranking", mem, ctx,
            f"The {subj} is the {ctx} ocean on Earth.", "medium", "Oceans"))

    country_continent = [
        ("Egypt","Africa","Asia"), ("Turkey","Asia","Europe"),
        ("Indonesia","Asia","Australia"), ("Panama","North America","South America"),
        ("Colombia","South America","North America"),
        ("Venezuela","South America","North America"),
    ]
    for subj, mem, ctx in country_continent:
        facts.append(make_fact("geography", subj, "primary continent", mem, ctx,
            f"{subj} is primarily located in {ctx}.", "medium", "Country continents"))

    return facts

def build_prompts(facts, templates, regime):
    random.seed(42)
    prompts = []
    for fact in facts:
        template = random.choice(templates)
        prompt_text = template.format(
            conflict_sentence=fact['conflict_sentence'],
            relation=fact['relation'],
            subject=fact['subject']
        )
        prompts.append({
            "prompt_id": f"p_{uuid.uuid4().hex[:10]}",
            "fact_id": fact['id'],
            "regime": regime,
            "prompt": prompt_text,
            "context_answer": fact['context_answer'],
            "memory_answer": fact['memory_answer'],
            "category": fact['category'],
            "difficulty": fact['difficulty'],
        })
    return prompts

def main():
    random.seed(42)
    print("Generating facts...")
    all_facts = []
    all_facts.extend(geography())
    all_facts.extend(chemistry())
    all_facts.extend(astronomy())
    all_facts.extend(biology())
    all_facts.extend(physics())
    all_facts.extend(history())
    all_facts.extend(literature())
    all_facts.extend(music())
    all_facts.extend(mathematics())
    all_facts.extend(science_misc())
    all_facts.extend(sports())
    all_facts.extend(geography_misc())

    # Deduplicate on subject + relation pair
    seen = set()
    facts = []
    for f in all_facts:
        key = f["subject"] + "__" + f["relation"]
        if key not in seen:
            seen.add(key)
            facts.append(f)

    print(f"Generated {len(facts)} unique real-world facts.")

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(f"{out_dir}/raw", exist_ok=True)
    os.makedirs(f"{out_dir}/generated", exist_ok=True)
    os.makedirs(f"{out_dir}/manifests", exist_ok=True)

    save_jsonl(f"{out_dir}/raw/facts.jsonl", facts)
    save_jsonl(f"{out_dir}/generated/explicit_context_prompts.jsonl",
               build_prompts(facts, EXPLICIT_CONTEXT_TEMPLATES, "explicit_context"))
    save_jsonl(f"{out_dir}/generated/explicit_memory_prompts.jsonl",
               build_prompts(facts, EXPLICIT_MEMORY_TEMPLATES, "explicit_memory"))
    save_jsonl(f"{out_dir}/generated/ambiguous_conflict_prompts.jsonl",
               build_prompts(facts, AMBIGUOUS_TEMPLATES, "ambiguous"))
    save_jsonl(f"{out_dir}/generated/naturalistic_rag_prompts.jsonl",
               build_prompts(facts, NATURALISTIC_RAG_TEMPLATES, "naturalistic"))

    cat_counts = {}
    for f in facts:
        cat_counts[f["category"]] = cat_counts.get(f["category"], 0) + 1

    manifest = {
        "num_facts": len(facts),
        "num_prompts_per_regime": len(facts),
        "total_prompts": len(facts) * 4,
        "category_distribution": cat_counts,
        "regimes": ["explicit_context", "explicit_memory", "ambiguous", "naturalistic"],
        "templates_used": {
            "explicit_context": len(EXPLICIT_CONTEXT_TEMPLATES),
            "explicit_memory": len(EXPLICIT_MEMORY_TEMPLATES),
            "ambiguous": len(AMBIGUOUS_TEMPLATES),
            "naturalistic": len(NATURALISTIC_RAG_TEMPLATES)
        },
        "random_seed": 42,
        "timestamp": datetime.datetime.now().isoformat(),
        "script_path": os.path.abspath(__file__),
        "note": "All facts are real-world. No synthetic filler. Both single-token and multi-token answers present by design."
    }
    save_json(f"{out_dir}/manifests/dataset_generation_manifest.json", manifest)
    print("Done. Category distribution:")
    for k, v in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
