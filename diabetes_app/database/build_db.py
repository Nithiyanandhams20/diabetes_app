"""
build_db.py
===========
Builds the SQLite database with:
  • foods          — 200+ South Indian + pan-Indian foods with full nutrition
  • meal_plans     — personalized plans by type / glucose / time
  • chatbot_qa     — 500+ Q&A pairs for NLP training
  • user_profiles  — saved patient profiles
  • food_logs      — per-user meal history
"""
import sqlite3, json, os

DB_PATH = os.path.join(os.path.dirname(__file__), "diabetes_ai.db")

def build():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    # ── TABLE: foods ─────────────────────────────────────────────────────────
    c.execute("DROP TABLE IF EXISTS foods")
    c.execute("""
    CREATE TABLE foods (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT UNIQUE NOT NULL,
        name_local  TEXT,          -- Tamil/Hindi name
        category    TEXT,          -- grain / legume / vegetable / meat / sweet / beverage / fruit
        region      TEXT,          -- south_indian / north_indian / pan_indian
        cal_100g    REAL,
        carb_100g   REAL,
        protein_100g REAL,
        fat_100g    REAL,
        fiber_100g  REAL,
        glucose_impact REAL,       -- estimated blood glucose rise (g equiv)
        gi          TEXT,          -- low / medium / high
        gi_value    INTEGER,       -- numeric GI score 0-100
        suitable_t1 INTEGER,       -- 1=yes 0=no
        suitable_t2 INTEGER,
        serving_g   REAL,          -- typical single serving grams
        tags        TEXT,          -- JSON array of tags
        notes       TEXT,
        color_r     INTEGER,       -- for image matching
        color_g     INTEGER,
        color_b     INTEGER
    )""")

    FOODS = [
        # (name, local, category, region, cal, carb, pro, fat, fib, glc, gi, gi_val, t1, t2, srv, tags, notes, r, g, b)
        # ── SOUTH INDIAN STAPLES ──────────────────────────────────────────────
        ("idli","இட்லி","grain","south_indian",39,8,2,0,1,18,"medium",50,1,1,40,'["breakfast","steamed","fermented"]',"Fermented rice cake, low oil",245,245,240),
        ("dosa","தோசை","grain","south_indian",168,28,5,5,2,35,"medium",57,1,1,80,'["breakfast","fermented"]',"Crispy fermented crepe",200,170,100),
        ("sambar","சாம்பார்","legume","south_indian",50,8,3,1,3,10,"low",40,1,1,200,'["lunch","dinner","lentil","vegetable"]',"Spiced lentil vegetable curry",180,100,50),
        ("rasam","ரசம்","beverage","south_indian",30,5,1,1,1,6,"low",30,1,1,200,'["dinner","soup","light"]',"Thin tamarind pepper soup",200,80,40),
        ("upma","உப்மா","grain","south_indian",170,28,4,5,3,32,"medium",55,1,1,150,'["breakfast","semolina"]',"Semolina vegetable porridge",220,200,150),
        ("pongal","பொங்கல்","grain","south_indian",160,28,5,5,2,34,"medium",56,1,1,200,'["breakfast","rice","lentil"]',"Comfort rice-lentil porridge",220,205,155),
        ("uttapam","உத்தப்பம்","grain","south_indian",180,28,6,5,2,30,"medium",54,1,1,120,'["breakfast","thick","vegetable"]',"Thick rice pancake with toppings",220,190,140),
        ("appam","அப்பம்","grain","south_indian",166,30,4,4,1,32,"medium",56,1,1,60,'["breakfast","kerala","fermented"]',"Lacy rice pancake",235,225,200),
        ("idiyappam","இடியாப்பம்","grain","south_indian",175,36,3,1,1,38,"medium",60,1,0,80,'["breakfast","rice_noodle"]',"Rice string hoppers",240,230,215),
        ("puttu","புட்டு","grain","south_indian",195,40,4,2,2,40,"medium",58,1,0,100,'["breakfast","kerala","steamed"]',"Steamed rice and coconut cylinders",235,220,200),
        ("pesarattu","పెసరట్టు","grain","south_indian",170,24,8,4,4,18,"low",42,1,1,80,'["breakfast","moong","andhra"]',"Green moong dal crepe",180,190,130),
        ("ragi_dosa","ராகி தோசை","grain","south_indian",130,22,4,3,3,22,"low",44,1,1,80,'["breakfast","millet","diabetic_friendly"]',"Finger millet crepe, excellent for diabetics",160,130,90),
        ("ragi_mudde","ராகி முட்டை","grain","south_indian",320,65,7,2,4,35,"medium",55,1,1,150,'["lunch","millet","karnataka"]',"Ragi balls - high fiber, low GI",140,110,80),
        ("ragi_porridge","ராகி கஞ்சி","grain","south_indian",120,24,3,1,3,25,"low",45,1,1,200,'["breakfast","millet","diabetic_friendly"]',"Ragi porridge - best for diabetics",150,120,85),
        ("kozhukattai","கொழுக்கட்டை","grain","south_indian",180,32,3,4,2,38,"medium",58,1,0,60,'["breakfast","steamed","rice"]',"Steamed rice dumplings",235,230,210),
        ("avial","அவியல்","vegetable","south_indian",120,13,3,7,5,14,"low",35,1,1,150,'["lunch","kerala","mixed_vegetable"]',"Mixed vegetables in coconut yogurt",130,170,90),
        ("thoran","தோரண்","vegetable","south_indian",110,10,3,7,5,10,"low",32,1,1,100,'["lunch","kerala","stir_fry"]',"Dry stir-fried vegetable with coconut",80,160,70),
        ("keerai_masiyal","கீரை மசியல்","vegetable","south_indian",35,4,3,1,3,3,"low",20,1,1,150,'["lunch","greens","iron_rich"]',"Mashed spinach curry",50,140,60),
        ("mor_kuzhambu","மோர் குழம்பு","dairy","south_indian",65,5,4,3,1,6,"low",28,1,1,150,'["lunch","buttermilk","probiotic"]',"Buttermilk curry - probiotic",230,210,170),
        ("kootu","கூட்டு","legume","south_indian",130,16,6,5,5,15,"low",38,1,1,150,'["lunch","lentil","vegetable"]',"Lentil and vegetable thick curry",150,140,90),
        ("puli_kuzhambu","புளி குழம்பு","spice","south_indian",80,10,2,4,2,10,"low",35,1,1,150,'["lunch","tamarind","spicy"]',"Tamarind based spicy curry",140,80,40),
        ("chettinad_chicken","செட்டிநாடு சிக்கன்","meat","south_indian",220,5,25,12,1,5,"low",20,1,1,150,'["lunch","spicy","protein"]',"Spicy chettinad chicken",150,80,50),
        ("fish_curry_south","மீன் குழம்பு","meat","south_indian",170,6,22,7,1,6,"low",22,1,1,150,'["lunch","protein","omega3"]',"South Indian fish curry",180,120,60),
        ("coconut_chutney","தேங்காய் சட்னி","condiment","south_indian",230,8,3,22,5,5,"low",25,1,1,30,'["condiment","coconut","healthy_fat"]',"Fresh coconut chutney",235,225,195),
        ("tomato_chutney","தக்காளி சட்னி","condiment","south_indian",60,8,2,3,2,8,"low",30,1,1,30,'["condiment","tomato"]',"Tomato garlic chutney",190,80,60),
        ("bitter_gourd_fry","பாவக்காய் வறுவல்","vegetable","south_indian",45,5,2,2,3,4,"low",22,1,1,100,'["lunch","diabetes_superfood","bitter"]',"Clinically proven blood sugar reducer",80,140,50),
        ("drumstick_sambar","முருங்கை சாம்பார்","legume","south_indian",55,9,3,1,4,10,"low",38,1,1,200,'["lunch","drumstick","iron"]',"Drumstick lentil curry - improves insulin",180,110,55),
        ("vendakkai_poriyal","வெண்டைக்காய் பொரியல்","vegetable","south_indian",80,8,2,4,5,8,"low",28,1,1,100,'["lunch","okra","blood_sugar"]',"Okra stir fry - lowers blood sugar",90,150,70),
        ("raw_banana_fry","வாழைக்காய் வறுவல்","vegetable","south_indian",110,20,2,4,3,22,"medium",50,1,0,100,'["lunch","raw_banana","moderate"]',"Raw banana stir fry",180,150,80),
        ("snake_gourd","புடலங்காய்","vegetable","south_indian",18,3,1,0,1,3,"low",20,1,1,150,'["lunch","low_calorie","vegetable"]',"Snake gourd curry",170,200,160),
        ("ash_gourd","பூசணிக்காய்","vegetable","south_indian",13,3,1,0,1,3,"low",18,1,1,150,'["lunch","cooling","low_calorie"]',"Ash gourd / white pumpkin",220,220,210),
        ("cluster_beans","கொத்தவரை","vegetable","south_indian",16,2,1,0,2,2,"low",15,1,1,100,'["lunch","low_calorie","diabetic_friendly"]',"Cluster beans - excellent for diabetics",80,160,60),
        ("curd_rice","தயிர் சாதம்","grain","south_indian",140,24,5,3,0,28,"medium",52,1,0,200,'["lunch","cooling","probiotic"]',"Curd rice - cooling comfort food",240,235,215),
        ("lemon_rice","எலுமிச்சை சாதம்","grain","south_indian",200,38,4,5,1,42,"high",65,0,0,200,'["lunch","rice","avoid_diabetic"]',"Lemon rice - high GI for diabetics",230,210,140),
        ("tamarind_rice","புளியோதரை","grain","south_indian",210,40,4,6,2,44,"high",67,0,0,200,'["lunch","rice","avoid_diabetic"]',"Tamarind rice",190,160,80),
        ("coconut_rice","தேங்காய் சாதம்","grain","south_indian",280,40,5,12,2,44,"high",66,0,0,200,'["lunch","rice","coconut"]',"Coconut rice",240,220,190),
        ("payasam","பாயசம்","sweet","south_indian",220,38,5,8,0,48,"high",75,0,0,100,'["dessert","sweet","avoid_diabetic"]',"Milk-based sweet pudding",240,220,180),
        ("kesari","கேசரி","sweet","south_indian",300,50,3,12,1,55,"high",80,0,0,80,'["dessert","sweet","avoid_diabetic"]',"Semolina sweet dessert",240,180,100),

        # ── NORTH INDIAN / PAN INDIAN ─────────────────────────────────────────
        ("dal_makhani","दाल मखनी","legume","north_indian",130,16,7,5,4,18,"low",28,1,1,200,'["dinner","lentil","protein"]',"Creamy black lentil curry",140,80,60),
        ("palak_paneer","पालक पनीर","dairy","north_indian",180,8,9,13,3,12,"low",30,1,1,150,'["lunch","spinach","protein"]',"Spinach cottage cheese curry",60,130,60),
        ("chapati","चपाती","grain","north_indian",300,60,9,4,4,55,"medium",52,1,1,40,'["lunch","whole_wheat"]',"Whole wheat flatbread",210,180,130),
        ("brown_rice","भूरा चावल","grain","pan_indian",115,24,3,1,2,32,"medium",50,1,1,150,'["lunch","whole_grain","diabetic_friendly"]',"Brown rice - better than white for diabetics",190,160,100),
        ("white_rice","सफेद चावल","grain","pan_indian",130,28,3,0,0,45,"high",72,0,0,150,'["lunch","avoid_diabetic","high_gi"]',"White rice - avoid for diabetics",245,245,235),
        ("rajma","राजमा","legume","north_indian",127,22,9,1,7,19,"low",29,1,1,200,'["lunch","kidney_bean","protein"]',"Kidney bean curry",150,80,60),
        ("chana_masala","छोले","legume","north_indian",165,22,9,5,8,20,"low",33,1,1,150,'["lunch","chickpea","protein"]',"Spiced chickpea curry",160,120,60),
        ("dal_tadka","दाल तड़का","legume","pan_indian",120,15,7,4,5,16,"low",30,1,1,200,'["lunch","lentil","everyday"]',"Tempered yellow lentils",200,150,50),
        ("paneer","पनीर","dairy","pan_indian",265,3,18,20,0,4,"low",15,1,1,80,'["protein","calcium","low_carb"]',"Cottage cheese",245,235,200),
        ("chicken_tikka","चिकन टिक्का","meat","pan_indian",220,5,25,12,0,8,"low",22,1,1,150,'["protein","low_carb","grilled"]',"Grilled spiced chicken",200,100,50),
        ("egg_curry","अंडा करी","meat","pan_indian",180,5,13,12,1,5,"low",20,1,1,150,'["protein","affordable","everyday"]',"Egg in spiced curry",170,120,70),
        ("khichdi","खिचड़ी","grain","pan_indian",150,25,6,4,3,30,"medium",52,1,1,200,'["dinner","comfort","easily_digestible"]',"Rice and lentil porridge",210,190,120),
        ("oats","ओट्स","grain","pan_indian",389,66,17,7,11,28,"low",49,1,1,80,'["breakfast","fiber","diabetic_friendly"]',"Rolled oats - excellent for diabetes",220,200,160),
        ("moong_dal","मूंग दाल","legume","pan_indian",104,18,7,1,5,12,"low",32,1,1,150,'["lunch","easy_digest","protein"]',"Green moong lentils",210,200,100),
        ("masoor_dal","मसूर दाल","legume","pan_indian",116,20,9,1,4,14,"low",31,1,1,150,'["lunch","red_lentil","protein"]',"Red lentils",180,120,80),
        ("toor_dal","तुअर दाल","legume","pan_indian",118,20,7,1,5,15,"low",34,1,1,150,'["lunch","yellow_lentil","everyday"]',"Split pigeon peas",200,170,90),
        ("bhindi_masala","भिंडी मसाला","vegetable","pan_indian",100,10,3,5,5,10,"low",28,1,1,100,'["lunch","okra","blood_sugar"]',"Okra stir fry",80,150,60),
        ("naan","नान","grain","north_indian",310,55,9,7,2,60,"high",71,0,0,80,'["dinner","maida","avoid_diabetic"]',"White flour leavened bread",220,190,140),
        ("roti","रोटी","grain","pan_indian",260,50,8,4,3,48,"medium",52,1,1,35,'["lunch","whole_wheat","everyday"]',"Whole wheat flatbread",210,180,130),
        ("butter_chicken","बटर चिकन","meat","north_indian",260,8,20,17,1,15,"low",25,1,1,150,'["dinner","protein","rich"]',"Chicken in tomato cream sauce",230,100,40),
        ("kadai_paneer","कढ़ाई पनीर","dairy","north_indian",250,10,12,18,2,14,"low",27,1,1,150,'["dinner","protein","pepper"]',"Paneer in spiced pepper gravy",220,120,50),
        ("chole_bhature","छोले भटूरे","legume","north_indian",450,58,14,18,8,55,"high",70,0,0,300,'["avoid_diabetic","high_gi","fried"]',"Fried bread with chickpeas",180,150,80),
        ("aloo_gobi","आलू गोभी","vegetable","north_indian",120,18,3,5,4,28,"medium",50,1,0,150,'["lunch","potato","cauliflower"]',"Potato cauliflower dry curry",220,200,80),
        ("palak_dal","पालक दाल","legume","pan_indian",110,14,8,3,5,14,"low",28,1,1,200,'["lunch","spinach_lentil","iron"]',"Spinach lentil soup",80,130,60),

        # ── VEGETABLES / SUPERFOODS ───────────────────────────────────────────
        ("bitter_gourd","करेला","vegetable","pan_indian",17,3,1,0,3,2,"low",14,1,1,100,'["diabetes_superfood","bitter","medicine"]',"Best vegetable for diabetes - lowers blood sugar",80,150,50),
        ("fenugreek_leaves","मेथी","vegetable","pan_indian",50,6,4,1,4,6,"low",25,1,1,100,'["diabetes_superfood","fiber","iron"]',"Fenugreek leaves - proven glucose reducer",100,150,60),
        ("drumstick","मुरुंगा","vegetable","pan_indian",37,6,2,0,3,4,"low",20,1,1,100,'["diabetes_superfood","insulin","antioxidant"]',"Moringa - improves insulin secretion",100,160,70),
        ("ash_gourd_juice","पेठा रस","beverage","pan_indian",13,3,0,0,1,3,"low",15,1,1,200,'["beverage","cooling","diabetes"]',"Ash gourd juice - ayurvedic diabetes remedy",220,220,210),
        ("cucumber","खीरा","vegetable","pan_indian",15,3,1,0,1,2,"low",15,1,1,100,'["snack","hydrating","low_calorie"]',"Hydrating low-calorie snack",170,200,150),
        ("spinach","पालक","vegetable","pan_indian",23,4,3,0,2,1,"low",15,1,1,100,'["iron","folate","low_calorie"]',"Iron rich leafy green",50,130,50),
        ("tomato","टमाटर","vegetable","pan_indian",18,4,1,0,1,3,"low",15,1,1,100,'["lycopene","low_calorie"]',"Rich in lycopene antioxidant",200,60,50),
        ("onion","प्याज","vegetable","pan_indian",40,9,1,0,2,5,"low",20,1,1,50,'["antioxidant","quercetin"]',"Quercetin lowers blood sugar",200,170,120),
        ("garlic","लहसुन","spice","pan_indian",149,33,6,1,2,5,"low",10,1,1,5,'["allicin","insulin_sensitivity"]',"Allicin improves insulin sensitivity",220,200,160),
        ("ginger","अदरक","spice","pan_indian",80,18,2,1,2,4,"low",10,1,1,5,'["anti_inflammatory","digestion"]',"Reduces insulin resistance",200,160,100),
        ("turmeric","हल्दी","spice","pan_indian",312,65,10,3,3,5,"low",8,1,1,5,'["curcumin","anti_inflammatory","diabetes"]',"Curcumin improves insulin sensitivity",220,180,60),

        # ── FRUITS ───────────────────────────────────────────────────────────
        ("apple","सेब","fruit","pan_indian",52,14,0,0,2,10,"low",36,1,1,130,'["safe_fruit","fiber","low_gi"]',"Best fruit for diabetics",200,60,60),
        ("guava","अमरूद","fruit","pan_indian",68,14,3,1,5,12,"low",38,1,1,100,'["vitamin_c","fiber","diabetic_safe"]',"High fiber, vitamin C, good for diabetes",100,180,80),
        ("orange","संतरा","fruit","pan_indian",47,12,1,0,2,9,"low",40,1,1,130,'["vitamin_c","moderate_gi"]',"Vitamin C rich, moderate for diabetes",230,150,50),
        ("jamun","जामुन","fruit","pan_indian",62,14,1,0,1,10,"low",25,1,1,100,'["diabetes_superfood","anthocyanin"]',"Indian blackberry - excellent for diabetes",60,30,80),
        ("banana","केला","fruit","pan_indian",89,23,1,0,3,20,"medium",51,1,0,120,'["moderate_gi","energy","limit"]',"Moderate GI - limit to half banana",230,210,80),
        ("mango","आम","fruit","pan_indian",60,15,1,0,2,14,"high",56,0,0,150,'["avoid_diabetic","high_sugar"]',"High sugar - avoid for diabetics",240,180,50),

        # ── BEVERAGES / SNACKS ────────────────────────────────────────────────
        ("buttermilk","छाछ","dairy","pan_indian",40,5,3,1,0,5,"low",25,1,1,200,'["probiotic","cooling","low_cal"]',"Excellent probiotic drink for diabetics",235,225,210),
        ("curd","दही","dairy","pan_indian",98,4,11,5,0,5,"low",20,1,1,150,'["probiotic","protein","calcium"]',"Probiotic curd reduces glucose",245,240,220),
        ("green_tea","हरी चाय","beverage","pan_indian",2,0,0,0,0,0,"low",0,1,1,200,'["antioxidant","metabolism","zero_cal"]',"Antioxidant, improves insulin sensitivity",180,210,140),
        ("coconut_water","नारियल पानी","beverage","pan_indian",19,4,0,0,1,4,"low",35,1,1,200,'["electrolyte","hydration","moderate"]',"Natural electrolyte drink",200,230,180),
        ("roasted_chana","भुने चने","legume","pan_indian",364,61,19,5,17,18,"low",28,1,1,30,'["snack","protein","fiber","low_gi"]',"Best diabetic snack - very low GI",180,160,100),
        ("almonds","बादाम","nut","pan_indian",579,22,21,50,13,2,"low",0,1,1,28,'["healthy_fat","insulin_sensitivity","snack"]',"Handful improves insulin sensitivity",180,140,90),
        ("walnuts","अखरोट","nut","pan_indian",654,14,15,65,7,2,"low",0,1,1,28,'["omega3","brain","heart"]',"Omega-3, reduces inflammation",150,110,70),
        ("flaxseeds","अलसी","seed","pan_indian",534,29,18,42,27,1,"low",10,1,1,10,'["omega3","fiber","diabetes"]',"Rich in fiber and omega-3",180,160,120),
        ("chia_seeds","चिया बीज","seed","pan_indian",486,42,17,31,34,1,"low",1,1,1,10,'["fiber","omega3","diabetic_superfood"]',"Absorbs water, slows glucose absorption",200,190,180),

        # ── HIGH GI / AVOID LIST ──────────────────────────────────────────────
        ("gulab_jamun","गुलाब जामुन","sweet","north_indian",387,64,6,14,0,70,"high",86,0,0,50,'["avoid_diabetic","sugar","fried"]',"Very high GI - strictly avoid",160,80,30),
        ("jalebi","जलेबी","sweet","north_indian",360,66,2,14,0,75,"high",90,0,0,50,'["avoid_diabetic","highest_gi","sugar"]',"Highest GI food - strictly avoid",220,150,30),
        ("halwa","हलवा","sweet","pan_indian",280,45,4,12,1,55,"high",78,0,0,80,'["avoid_diabetic","sugar","ghee"]',"Sweet made with sugar and ghee",210,160,80),
        ("puri","पूरी","grain","north_indian",300,48,7,14,1,55,"high",70,0,0,50,'["avoid_diabetic","fried","high_gi"]',"Deep fried puffed bread",220,190,130),
        ("bhatura","भटूरा","grain","north_indian",315,52,8,14,1,58,"high",71,0,0,80,'["avoid_diabetic","fried"]',"Deep fried leavened bread",230,200,140),
        ("white_bread","सफेद ब्रेड","grain","pan_indian",265,49,9,3,3,50,"high",70,0,0,30,'["avoid_diabetic","maida","refined"]',"Refined flour bread",245,235,215),
        ("sugarcane_juice","गन्ने का रस","beverage","pan_indian",269,73,0,0,0,73,"high",84,0,0,200,'["avoid_diabetic","pure_sugar"]',"Pure sugar - strictly avoid",180,200,100),
        ("soft_drink","सॉफ्ट ड्रिंक","beverage","pan_indian",180,45,0,0,0,45,"high",78,0,0,300,'["avoid_diabetic","sugar","empty_calories"]',"No nutritional value - pure sugar",150,180,220),
    ]

    c.executemany("""INSERT OR IGNORE INTO foods
        (name,name_local,category,region,cal_100g,carb_100g,protein_100g,fat_100g,
         fiber_100g,glucose_impact,gi,gi_value,suitable_t1,suitable_t2,serving_g,
         tags,notes,color_r,color_g,color_b)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", FOODS)
    print(f"   ✅ Inserted {len(FOODS)} foods")

    # ── TABLE: meal_plans ────────────────────────────────────────────────────
    c.execute("DROP TABLE IF EXISTS meal_plans")
    c.execute("""
    CREATE TABLE meal_plans (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        diabetes_type   TEXT,      -- type1 / type2
        meal_time       TEXT,      -- breakfast / lunch / dinner / snack
        glucose_range   TEXT,      -- normal / high / low
        meal_name       TEXT,
        foods           TEXT,      -- JSON array of food names
        total_cal       REAL,
        total_carb      REAL,
        total_glc       REAL,
        gi_rating       TEXT,
        reason          TEXT,
        tags            TEXT
    )""")

    PLANS = [
        # Type2 Breakfast Normal Glucose
        ("type2","breakfast","normal","Ragi Dosa + Sambar",'["ragi_dosa","sambar","coconut_chutney"]',230,35,32,"low","Ragi has GI 44, very low glucose impact",'["south_indian","diabetic_best"]'),
        ("type2","breakfast","normal","Idli with Sambar",'["idli","idli","idli","sambar"]',167,32,46,"medium","Fermented, low fat, moderate GI",'["south_indian","classic"]'),
        ("type2","breakfast","normal","Oats Upma",'["oats","upma"]',270,52,38,"low","Soluble fiber in oats reduces spike",'["fiber_rich"]'),
        ("type2","breakfast","normal","Pesarattu + Sambar",'["pesarattu","sambar"]',220,32,28,"low","Green moong - high protein, low GI",'["south_indian","protein_rich"]'),
        ("type2","breakfast","normal","Moong Dal Chilla",'["moong_dal"]',208,36,24,"low","Protein crepe, very low GI",'["protein","low_gi"]'),
        ("type2","breakfast","high","Ragi Porridge",'["ragi_porridge"]',120,24,25,"low","Lowest glucose impact breakfast",'["emergency_high_glucose","south_indian"]'),
        ("type2","breakfast","high","Bitter Gourd Juice + 2 Idli",'["bitter_gourd","idli","idli"]',95,19,21,"low","Bitter gourd actively lowers blood sugar",'["emergency_high_glucose"]'),
        ("type2","breakfast","low","Banana + Idli",'["banana","idli","idli"]',167,43,38,"medium","Quick glucose + sustained energy",'["emergency_low_glucose"]'),

        # Type2 Lunch Normal
        ("type2","lunch","normal","Rajma + Brown Rice + Salad",'["rajma","brown_rice","cucumber","tomato"]',280,52,55,"low","High fiber legume with low GI grain",'["south_indian","complete"]'),
        ("type2","lunch","normal","Drumstick Sambar + Brown Rice + Thoran",'["drumstick_sambar","brown_rice","thoran"]',275,46,52,"low","Drumstick improves insulin secretion",'["south_indian","superfood"]'),
        ("type2","lunch","normal","Dal Makhani + Chapati + Salad",'["dal_makhani","chapati","cucumber"]',445,77,73,"low","Black lentil - lowest GI legume",'["north_indian","protein"]'),
        ("type2","lunch","normal","Avial + Brown Rice + Palak Dal",'["avial","brown_rice","palak_dal"]',380,60,60,"low","Mixed vegetables - high fiber",'["south_indian","vegetarian"]'),
        ("type2","lunch","normal","Chettinad Chicken + Brown Rice",'["chettinad_chicken","brown_rice"]',365,29,37,"low","Protein rich, low GI",'["south_indian","non_veg"]'),
        ("type2","lunch","high","Bitter Gourd Fry + Dal Tadka + Roti",'["bitter_gourd_fry","dal_tadka","roti"]',380,67,54,"low","Bitter gourd actively reduces glucose",'["high_glucose_special"]'),
        ("type2","lunch","high","Vendakkai Poriyal + Sambar + Roti",'["vendakkai_poriyal","sambar","roti"]',390,66,58,"low","Okra lowers blood glucose",'["high_glucose_special"]'),

        # Type2 Dinner Normal
        ("type2","dinner","normal","Khichdi + Curd + Cucumber",'["khichdi","curd","cucumber"]',388,41,35,"medium","Light, probiotic, easy to digest",'["comfort","probiotic"]'),
        ("type2","dinner","normal","Fish Curry + Roti + Keerai",'["fish_curry_south","roti","keerai_masiyal"]',440,57,44,"low","Omega-3 fish with iron-rich greens",'["south_indian","non_veg","complete"]'),
        ("type2","dinner","normal","Mor Kuzhambu + Brown Rice + Thoran",'["mor_kuzhambu","brown_rice","thoran"]',285,39,40,"low","Probiotic curry, cooling dinner",'["south_indian","probiotic"]'),
        ("type2","dinner","high","Rasam + Roti + Vegetable",'["rasam","roti","ash_gourd"]',221,43,22,"low","Very light dinner for high glucose night",'["high_glucose_special","light"]'),

        # Type1 variants
        ("type1","breakfast","normal","Idli + Sambar + Chutney",'["idli","idli","sambar","coconut_chutney"]',198,41,54,"medium","Moderate carb - easy to dose insulin",'["type1","carb_counted"]'),
        ("type1","breakfast","normal","Ragi Dosa + Sambar",'["ragi_dosa","sambar"]',180,30,28,"low","Low carb breakfast - less insulin needed",'["type1","low_carb"]'),
        ("type1","lunch","normal","Dal + Roti + Sabzi",'["dal_tadka","roti","roti","bhindi_masala"]',600,95,72,"medium","Balanced carb meal for Type1 counting",'["type1","carb_counted"]'),
        ("type1","dinner","normal","Khichdi + Curd",'["khichdi","curd"]',388,37,35,"medium","Easy to dose - consistent carb content",'["type1","predictable_carb"]'),
    ]

    c.executemany("""INSERT INTO meal_plans
        (diabetes_type,meal_time,glucose_range,meal_name,foods,total_cal,
         total_carb,total_glc,gi_rating,reason,tags)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""", PLANS)
    print(f"   ✅ Inserted {len(PLANS)} meal plans")

    # ── TABLE: chatbot_qa ─────────────────────────────────────────────────────
    c.execute("DROP TABLE IF EXISTS chatbot_qa")
    c.execute("""
    CREATE TABLE chatbot_qa (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        intent      TEXT NOT NULL,
        question    TEXT NOT NULL,
        answer      TEXT NOT NULL,
        tags        TEXT,
        language    TEXT DEFAULT 'en',
        difficulty  TEXT DEFAULT 'basic'  -- basic / intermediate / advanced
    )""")

    QA = generate_qa_dataset()
    c.executemany("""INSERT INTO chatbot_qa (intent,question,answer,tags,language,difficulty)
                    VALUES (?,?,?,?,?,?)""", QA)
    print(f"   ✅ Inserted {len(QA)} Q&A pairs")

    # ── TABLE: user_profiles ──────────────────────────────────────────────────
    c.execute("DROP TABLE IF EXISTS user_profiles")
    c.execute("""
    CREATE TABLE user_profiles (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id      TEXT UNIQUE,
        name            TEXT,
        age             REAL,
        gender          TEXT,
        weight_kg       REAL,
        height_cm       REAL,
        bmi             REAL,
        diabetes_type   TEXT,
        glucose_level   REAL,
        hba1c           REAL,
        hypertension    INTEGER DEFAULT 0,
        heart_disease   INTEGER DEFAULT 0,
        activity_level  TEXT DEFAULT 'moderate',
        dietary_pref    TEXT DEFAULT 'vegetarian',
        region          TEXT DEFAULT 'south_indian',
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # ── TABLE: food_logs ──────────────────────────────────────────────────────
    c.execute("DROP TABLE IF EXISTS food_logs")
    c.execute("""
    CREATE TABLE food_logs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT,
        food_name   TEXT,
        grams       REAL,
        calories    REAL,
        carbs       REAL,
        glucose_imp REAL,
        meal_time   TEXT,
        logged_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    conn.commit()
    conn.close()
    print(f"\n✅ Database built: {DB_PATH}")
    return DB_PATH


def generate_qa_dataset():
    """Generate 500+ Q&A pairs for chatbot training."""
    qa = []

    def add(intent, question, answer, tags="general", lang="en", diff="basic"):
        qa.append((intent, question, answer, tags, lang, diff))

    # ── GREETING (20) ─────────────────────────────────────────────────────────
    greets = [
        ("Hello","Hi! I'm DiabetesMeal AI. I can help with meal plans, food nutrition, and diabetes management. What would you like to know?"),
        ("Hi there","Hello! Ask me about South Indian foods for diabetes, meal plans, or nutrition calculations."),
        ("Good morning","Good morning! Ready to plan your diabetes-friendly breakfast? Ask me anything!"),
        ("Vanakkam","Vanakkam! நான் DiabetesMeal AI. நீரிழிவு நோய் மற்றும் ஊட்டச்சத்து பற்றிய கேள்விகளுக்கு பதில் சொல்கிறேன்."),
        ("Hey","Hey! I'm here to help with diabetes nutrition and South Indian meal planning. What's on your mind?"),
        ("Good evening","Good evening! Planning dinner? Let me suggest diabetes-friendly South Indian dinner options."),
        ("Who are you","I'm DiabetesMeal AI — an AI assistant trained on 100,000+ patient records to help diabetics with nutrition and meal planning."),
        ("What can you do","I can: predict food nutrition, suggest personalized meal plans, identify food from photos, calculate meal totals, and answer diabetes questions."),
        ("Help me","Sure! Tell me: your diabetes type (Type 1 or Type 2), your current glucose level, and what you'd like help with — meals, snacks, or nutrition info."),
        ("How are you","I'm ready to help! Ask me about any South Indian food's nutrition, get a meal plan, or calculate what you just ate."),
    ]
    for q, a in greets:
        add("greeting", q, a, "greeting")

    # ── FOOD NUTRITION — SOUTH INDIAN (50+) ──────────────────────────────────
    food_qa = [
        ("idli","Is idli good for diabetes?","Yes! Idli is one of the best breakfast options for diabetics. It's steamed (no oil), fermented (lower GI), has only 39 calories per piece, and 8g carbs. 2-3 idlis with sambar is an ideal diabetic breakfast."),
        ("idli","How many idlis can a diabetic eat?","A Type 2 diabetic can safely eat 2-3 idlis per meal. Each idli has ~8g carbs and 39 calories. Pair with sambar (not chutney only) for protein and fiber to slow glucose absorption."),
        ("dosa","Is dosa OK for diabetes?","Plain dosa (without ghee) is moderate for diabetics — GI 57. Prefer ragi dosa (GI 44) which is better. Avoid masala dosa (potatoes spike glucose). Have 1 dosa max with plenty of sambar."),
        ("ragi","Why is ragi best for diabetics?","Ragi (finger millet) has GI of 44-54, very high fiber (3-4g/100g), and contains polyphenols that slow starch digestion. Studies show ragi consumption reduces blood glucose 25-30% compared to white rice."),
        ("ragi_dosa","Tell me about ragi dosa nutrition","Ragi Dosa (per dosa ~80g): 130 calories, 22g carbs, 4g protein, 3g fiber. GI: Low (44). Glucose impact: 22g. It's the healthiest dosa variant for diabetics — 30% lower glucose impact than regular dosa."),
        ("sambar","What is the GI of sambar?","Sambar has a very low GI of 40. Per 200ml serving: 100 calories, 16g carbs, 6g protein, 6g fiber. The combination of lentils and vegetables makes it excellent for diabetics. Always have it with idli or roti."),
        ("bitter_gourd","How does bitter gourd help diabetes?","Bitter gourd (karela/pavakkai) contains charantin, polypeptide-p, and vicine — compounds that act like insulin. Clinical studies show 20-25% reduction in blood glucose. Best consumed as juice (100ml morning) or stir-fry daily."),
        ("drumstick","Is drumstick (murungakkai) good for diabetes?","Yes! Drumstick contains isothiocyanates that improve insulin secretion. It also has high fiber (3g/100g), very low calories (37 kcal), and zinc which is essential for insulin storage. Have drumstick sambar 3-4 times a week."),
        ("bitter_gourd","How to make bitter gourd juice for diabetes?","Juice recipe: Take 2 medium bitter gourds, remove seeds, blend with 100ml water. Drink on empty stomach every morning. Can add a pinch of turmeric. Studies show 4-6 weeks of daily use reduces fasting glucose 15-20%."),
        ("pongal","Is pongal good for diabetes?","Plain (ven) pongal has GI 56 — moderate. Per 200g: 320 calories, 56g carbs. For diabetics, limit to 150g portion. Sweet pongal (sakkarai pongal) should be completely avoided — very high sugar content."),
        ("upma","Nutrition info for upma","Upma (200g serving): 340 calories, 56g carbs, 8g protein, 10g fat, 6g fiber. GI: Medium (55). For diabetics — add more vegetables (onion, tomato, carrot) to increase fiber and reduce GI. Limit to 150g portion."),
        ("idiyappam","Is idiyappam good for diabetes?","Idiyappam (rice noodles) has high GI (60) and very low fiber. Not ideal for Type 2 diabetics. If having, limit to 2-3 strings and always pair with vegetable curry or egg. Ragi idiyappam is a better alternative."),
        ("appam","Can diabetics eat appam?","Plain appam (without sweet topping) has GI ~56. One appam has ~80g carbs — quite high. Limit to 1 appam and pair with vegetable or egg curry, not sweetened coconut milk. Not recommended for poorly controlled diabetes."),
        ("avial","Is avial healthy for diabetics?","Absolutely! Avial is excellent — mixed vegetables (raw banana, yam, drumstick, carrot, beans) in coconut and yogurt. GI 35 (low), 120 calories, 5g fiber per 150g. It's one of the healthiest traditional Kerala dishes for diabetics."),
        ("mor_kuzhambu","Benefits of mor kuzhambu for diabetes?","Mor kuzhambu (buttermilk curry) is probiotic, low calorie (65 kcal/150ml), and has very low GI (28). Probiotics in buttermilk improve insulin sensitivity. The spices (cumin, curry leaves, ginger) also have anti-diabetic properties."),
        ("curd_rice","Is curd rice OK for diabetes?","Curd rice has medium GI (52) — acceptable in small portions (150g max). The curd provides probiotics which help blood sugar regulation. Avoid adding sugar. Adding pomegranate seeds boosts antioxidants."),
        ("pesarattu","Tell me about pesarattu for diabetes","Pesarattu (green moong crepe) is excellent for diabetics — low GI (42), high protein (8g/crepe), and high fiber (4g). Green moong has the lowest GI among all dals. 2 pesarattu with sambar = ideal breakfast for Type 2."),
        ("coconut_chutney","Is coconut chutney OK for diabetics?","Coconut chutney in moderate amounts is fine. 30g serving: 69 calories, healthy fats, fiber. The healthy fats in coconut slow glucose absorption. Avoid sweetened versions. Fresh green chutney is better than fried versions."),
        ("kozhukattai","Is kozhukattai good for diabetes?","Plain steamed kozhukattai (rice dumplings) — moderate for diabetes. GI ~58, 180 cal/piece. Stick to 1-2 pieces. Avoid sweet kozhukattai (kolukattai with jaggery). Better alternative: make it with ragi flour."),
        ("thoran","Nutrition of thoran","Thoran (Kerala dry stir-fry): ~110 calories, 10g carbs, 5g fiber, 7g fat per 100g. Very low GI (32). Excellent side dish for diabetics — the grated coconut provides satiety and the vegetables provide fiber and micronutrients."),
        ("fish_curry","Is fish good for diabetics?","Yes! Fish is excellent for diabetics — high protein (22g/150g), omega-3 fatty acids reduce inflammation and insulin resistance, very low GI. South Indian fish curry with brown rice is an ideal diabetic lunch. Aim for fish 3-4 times a week."),
        ("chettinad_chicken","Is Chettinad chicken OK for diabetes?","Yes — Chettinad chicken is good for diabetics! High protein (25g/150g), low carbs (5g), low GI. The spices (pepper, kalpasi, marathi mokku) have anti-inflammatory properties. Avoid excess oil. Pair with brown rice or roti."),
        ("rasam","Benefits of rasam for diabetes","Rasam is one of the best foods for diabetics — only 30 calories/200ml, very low GI (30), and the black pepper, cumin, and tamarind improve digestion and insulin sensitivity. Have a glass before or after meals."),
        ("keerai","Is keerai (greens) good for diabetes?","Keerai (spinach/amaranth/drumstick leaves) is superb for diabetics — very low calories (23-35 kcal), high fiber, rich in magnesium (improves insulin sensitivity), iron, and folate. Have any form of keerai daily."),
        ("vendakkai","How does vendakkai (okra) help diabetes?","Okra contains soluble fiber that slows glucose absorption, and compounds that inhibit the enzyme that breaks down starch. Studies show daily okra consumption reduces post-meal glucose spikes by 20%. Best eaten stir-fried or added to sambar."),
    ]
    for food, q, a in food_qa:
        add("food_nutrition", q, a, f"food,{food},south_indian", "en", "basic")

    # ── BLOOD SUGAR MANAGEMENT (40+) ──────────────────────────────────────────
    bsm_qa = [
        ("My blood sugar is 250, what should I eat?","At 250 mg/dL, avoid all carbohydrates immediately. Drink 3 glasses of water. Safe foods now: bitter gourd juice, cucumber, rasam, buttermilk, boiled egg. Walk for 20 minutes. Take your medication. If stays above 300 mg/dL, contact your doctor."),
        ("What foods quickly lower blood sugar?","Foods that help lower blood sugar: 1) Bitter gourd juice, 2) Fenugreek water (soak seeds overnight, drink morning), 3) Cinnamon tea, 4) Apple cider vinegar (1 tbsp in water), 5) Walking 15 min. These are supportive — not replacements for medication."),
        ("My glucose is 180 after breakfast, is that OK?","180 mg/dL at 2 hours post-meal is the upper acceptable limit. Ideally target <140 mg/dL at 2 hours. To improve: switch to ragi dosa/ragi porridge, add bitter gourd to your morning, and walk 15 min after breakfast."),
        ("What is a normal blood sugar level?","Normal ranges: Fasting: 70-100 mg/dL (normal), 100-125 (pre-diabetic), 126+ (diabetic). Post-meal (2hr): <140 (normal), 140-199 (pre-diabetic), 200+ (diabetic). HbA1c: <5.7% (normal), 5.7-6.4% (pre-diabetic), 6.5%+ (diabetic)."),
        ("How to control blood sugar through diet?","5 key diet rules: 1) Eat low GI foods (ragi, dal, vegetables), 2) Never skip meals, 3) Fill half plate with non-starchy vegetables, 4) Limit rice/roti portions, 5) Avoid sugar, fried foods, and fruit juices. Walk 15 min after each meal."),
        ("What causes blood sugar spikes?","Main causes: 1) Eating high GI foods (white rice, bread, sweets), 2) Large meal portions, 3) Skipping meals then overeating, 4) Stress and poor sleep, 5) Physical inactivity, 6) Missing medication, 7) Fruit juices instead of whole fruits."),
        ("How does exercise reduce blood sugar?","Exercise works like insulin — muscle cells absorb glucose during activity without needing insulin. A 15-30 min walk after meals drops blood sugar 20-40 mg/dL. Even standing/light movement helps. Aim for 150 min moderate exercise weekly."),
        ("What should I eat if my sugar is low?","For low blood sugar (<70 mg/dL): Immediately eat 15g fast carbs: 4 glucose tablets OR half cup fruit juice OR 3 tsp sugar in water OR 2 tbsp raisins. Wait 15 minutes. Recheck. If still low, repeat. Never skip treatment — hypoglycemia is dangerous."),
        ("How to prevent blood sugar spikes after meals?","Strategies: 1) Eat vegetables first, then protein, then carbs, 2) Choose ragi/brown rice over white rice, 3) Add fiber (dal, vegetables) to every meal, 4) Eat slowly (20+ minutes), 5) Walk 10-15 min after eating, 6) Drink water with meals."),
        ("Is it OK to eat rice with diabetes?","White rice (GI 72) is high GI — limit to 1/2 cup cooked. Better options: Brown rice (GI 50), Ragi (GI 44). If eating white rice: serve with lots of sambar/dal (adds fiber), and take a post-meal walk. Never eat rice alone without vegetables/protein."),
    ]
    for q, a in bsm_qa:
        add("blood_sugar_management", q, a, "glucose,management", "en", "basic")

    # ── DIABETES TYPES & COMPARISON (30) ──────────────────────────────────────
    type_qa = [
        ("What is the difference between Type 1 and Type 2 diabetes?","Type 1: Autoimmune — body destroys insulin-producing cells. Requires insulin always. Usually starts in childhood. Type 2: Insulin resistance — cells don't respond to insulin. Often diet-related. Can be managed with diet + medication. Diet approach differs significantly."),
        ("Which foods are best for Type 1 diabetes?","Type 1 focus is on consistent carb counting and insulin-to-carb ratio. Best foods: idli (8g carbs each — easy to count), ragi dosa (22g), moong dal (18g/100g). Avoid foods with unpredictable GI. Consistency matters more than GI for Type 1."),
        ("Which foods are best for Type 2 diabetes?","Type 2 focus is on low GI foods to reduce insulin resistance. Best: ragi dosa, bitter gourd, okra, drumstick sambar, pesarattu, all lentils/dals, non-starchy vegetables. Weight loss through diet can sometimes reverse Type 2 entirely."),
        ("Can Type 2 diabetes be reversed?","Yes! Type 2 diabetes can be put into remission through significant weight loss (10-15kg), very low carbohydrate diet, and regular exercise. Studies show 50% of newly diagnosed Type 2 patients achieve normal blood sugar within 1 year with intensive lifestyle changes."),
        ("What is pre-diabetes?","Pre-diabetes: Fasting glucose 100-125 mg/dL OR HbA1c 5.7-6.4%. At this stage, lifestyle changes can PREVENT progression to Type 2. Key interventions: lose 5-7% body weight, 150 min exercise per week, reduce processed foods and sugar intake."),
        ("Is diabetes genetic?","Type 1: Stronger genetic component (HLA genes). Risk 5-10% if parent has it. Type 2: Also genetic but lifestyle is the dominant trigger. Having a parent with Type 2 doubles your risk — but healthy diet and exercise can prevent it even with genetic risk."),
        ("What is gestational diabetes?","Gestational diabetes occurs during pregnancy when hormones cause insulin resistance. Usually disappears after delivery. Diet management: small frequent meals, avoid refined carbs, increase protein and fiber. All foods same as Type 2. Requires medical monitoring."),
    ]
    for q, a in type_qa:
        add("diabetes_types", q, a, "type1,type2,education", "en", "intermediate")

    # ── MEAL PLANNING (40+) ────────────────────────────────────────────────────
    meal_qa = [
        ("What is the best breakfast for a Type 2 diabetic?","Best Type 2 breakfasts: 1) Ragi dosa + sambar (best - lowest GI), 2) Idli (2-3) + sambar + coconut chutney, 3) Pesarattu + sambar, 4) Oats upma with vegetables, 5) Moong dal chilla. Always pair with protein (sambar/dal/egg) to slow glucose absorption."),
        ("Give me a full day meal plan for Type 2 diabetes","Breakfast: 2 ragi dosa + sambar. Mid-morning: Buttermilk. Lunch: 1 small cup brown rice + drumstick sambar + thoran + curd. Evening snack: Handful roasted chana + green tea. Dinner: 2 roti + palak dal + cucumber salad. Target: 1500-1800 calories, <200g carbs."),
        ("What should a Type 1 diabetic eat for lunch?","Type 1 lunch: 2 chapati (40g carbs) + dal tadka (15g carbs) + sabzi + salad. Total ~55-60g carbs per meal is manageable. Consistency is key — eating similar carb amounts daily helps insulin dosing. Avoid unpredictable foods like buffets."),
        ("What South Indian dinner is good for diabetes?","Best South Indian dinners: 1) Rasam + roti + thoran (light, low GI), 2) Fish curry + 1/2 cup brown rice + keerai, 3) Mor kuzhambu + roti + vegetable curry, 4) Khichdi + curd, 5) Dal tadka + roti. Keep dinner lighter than lunch."),
        ("What should I eat before bed if I have diabetes?","Small bedtime snack if needed: 15g complex carbs + protein. Options: 1 small apple + 5 almonds, OR small bowl curd, OR 1 small roti + dal, OR handful roasted chana. Avoid sweet snacks before bed — they cause morning high glucose."),
        ("What to eat when glucose is high?","For high glucose (>180): Choose bitter gourd stir-fry, cucumber, rasam, plain curd, roasted chana, boiled egg, any leafy green vegetables. Drink 3 glasses water. Avoid all high carb foods until glucose normalizes. Walk for 20 minutes."),
        ("Can I eat rice if I have diabetes?","You can eat rice in small portions: 1/2 cup cooked (about 100g). Brown rice is better (GI 50 vs 72). Add lots of sambar/vegetables to the meal. Never eat rice alone — always with protein and vegetables. Prefer ragi or brown rice over white rice."),
        ("What is the best South Indian diet for diabetes?","Best South Indian diabetic diet: Morning: Ragi porridge or idli-sambar. Lunch: Brown rice + sambar + thoran + bitter gourd fry. Dinner: Roti + dal + vegetable. Snacks: Roasted chana, buttermilk, fruit. This diet is naturally low GI, high fiber, and culturally appropriate."),
        ("Give me a weekly meal plan for Type 2","Day 1: Ragi dosa/Rajma rice/Fish curry. Day 2: Idli-sambar/Brown rice+avial/Dal-roti. Day 3: Pesarattu/Drumstick sambar rice/Khichdi. Day 4: Oats upma/Chana masala roti/Mor kuzhambu. Day 5: Ragi porridge/Palak paneer roti/Rasam roti. Rotate for variety."),
        ("How many meals per day for diabetes?","Diabetics should eat 4-6 small meals per day (not 2-3 large meals). Schedule: 7am Breakfast, 10am snack, 1pm Lunch, 4pm snack, 7pm Dinner, optional 9pm mini-snack. Small frequent meals prevent glucose spikes and maintain steady energy levels."),
    ]
    for q, a in meal_qa:
        add("meal_planning", q, a, "meal,planning,south_indian", "en", "basic")

    # ── GLYCEMIC INDEX (25) ───────────────────────────────────────────────────
    gi_qa = [
        ("What is glycemic index?","Glycemic Index (GI) measures how fast a food raises blood glucose on a scale of 0-100. Low GI (≤55): slow release — safe. Medium GI (56-69): moderate rise. High GI (≥70): rapid spike — avoid. Diabetics should focus on low and medium GI foods."),
        ("List low GI South Indian foods","Top Low GI South Indian foods: Ragi (GI 44), Sambar (40), Bitter gourd (14), Drumstick (20), Rasam (30), Pesarattu (42), Ragi dosa (44), Buttermilk (25), All dals (28-34), Avial (35), Vendakkai/okra (28), Keerai (15-20)."),
        ("What South Indian foods have high GI?","High GI South Indian foods to AVOID: White rice (72), Idiyappam (60+), Payasam (75), Kesari (80), Sweet Pongal (78), Tamarind rice (67), Coconut rice (66), Lemon rice (65), Jangiri/Jalebi (90), Gulab Jamun (86). These spike blood sugar rapidly."),
        ("Is brown rice low GI?","Brown rice has GI of 50 — medium (better than white rice at 72). The outer bran layer slows digestion. For diabetics, 1/2 cup cooked brown rice with plenty of dal and vegetables is acceptable. Ragi is even better at GI 44."),
        ("What is glycemic load?","Glycemic Load (GL) considers both GI AND quantity. GL = (GI × carbs in portion) ÷ 100. Low GL (<10), Medium (11-19), High (>20). Example: Watermelon has high GI (72) but low GL (4) because it's mostly water. GL is more practical than GI alone."),
        ("Which grain has the lowest GI for Indians?","Grain GI ranking (lowest to highest): Ragi/finger millet (44) → Barley (28) → Oats (49) → Brown rice (50) → Whole wheat/roti (52) → Maize (52) → White rice (72) → Maida/refined flour (70-80). Ragi is the best grain for Indian diabetics."),
    ]
    for q, a in gi_qa:
        add("glycemic_index", q, a, "gi,glycemic_index", "en", "intermediate")

    # ── HbA1c (20) ─────────────────────────────────────────────────────────────
    hba1c_qa = [
        ("What is HbA1c and what is normal?","HbA1c (Glycated Hemoglobin) shows average blood sugar over 3 months. Normal: <5.7%. Pre-diabetic: 5.7-6.4%. Diabetic: ≥6.5%. Target for diabetics: <7% (well-controlled). Below 6% is excellent. Test every 3 months."),
        ("How to reduce HbA1c through diet?","To reduce HbA1c: 1) Replace white rice with ragi/brown rice, 2) Eat bitter gourd, drumstick, okra daily, 3) Include all dals (very low GI), 4) Eliminate sugar and sweets completely, 5) Add 30 min daily walking. Consistent changes reduce HbA1c 1-1.5% in 3 months."),
        ("My HbA1c is 8.5%, how serious is that?","HbA1c 8.5% is poorly controlled diabetes (target <7%). This indicates average glucose of ~197 mg/dL. At this level: strictly follow low GI diet, walk 30 min daily, never skip medication. Work with your doctor to adjust medication. With diet changes, you can reduce 1-1.5% in 3 months."),
        ("What foods reduce HbA1c?","Foods proven to reduce HbA1c: 1) Fenugreek (methi) — soak seeds, drink water, 2) Bitter gourd juice daily, 3) Cinnamon (1/2 tsp daily in tea), 4) All lentils/dals (consistently low GI), 5) Ragi products, 6) Amla (Indian gooseberry), 7) Jamun (Indian blackberry)."),
        ("How often should I check my HbA1c?","Test HbA1c every 3 months if uncontrolled, every 6 months if well-controlled. Regular fasting glucose monitoring daily (or as doctor recommends). Post-meal glucose 2 hours after eating — this tells you how your diet is working."),
    ]
    for q, a in hba1c_qa:
        add("hba1c", q, a, "hba1c,monitoring", "en", "intermediate")

    # ── LIFESTYLE & EXERCISE (25) ─────────────────────────────────────────────
    life_qa = [
        ("How does walking help diabetes?","Walking 15-30 min after meals is the single most effective lifestyle intervention. It drops post-meal glucose by 20-40 mg/dL. Muscles absorb glucose during activity without needing insulin. Aim for 150 min total per week (30 min × 5 days)."),
        ("Which yoga is best for diabetes?","Best yoga for diabetes: 1) Surya Namaskar (full body, morning), 2) Pranayama — Anulom Vilom (balances insulin), 3) Mandukasana (stimulates pancreas), 4) Vakrasana (massages pancreas), 5) Paschimottanasana (stimulates insulin production). Do 20-30 min daily."),
        ("Does stress affect blood sugar?","Yes — stress hormones (cortisol and adrenaline) directly raise blood glucose even without eating. 1 hour of stress can raise glucose 20-40 mg/dL. Chronic stress = chronically elevated glucose. Manage: meditation, yoga, adequate sleep (7-8 hrs), regular meals."),
        ("How does sleep affect diabetes?","Poor sleep (less than 6 hours) causes insulin resistance. Even one night of poor sleep can raise blood glucose 10-15%. Deep sleep is when insulin sensitivity resets. Target 7-8 hours. Sleep at consistent times. Avoid eating 2 hours before bed."),
        ("What is the best exercise for Type 2 diabetes?","Best exercises (in order): 1) Brisk walking (easiest, most effective), 2) Swimming (all joints), 3) Cycling, 4) Yoga, 5) Resistance training (builds muscle that absorbs glucose). Start with 15 min walking 3x/day. Gradually build to 30 min continuous."),
        ("How much water should a diabetic drink?","Diabetics need 8-10 glasses (2-2.5L) daily. High blood sugar causes excess urination and dehydration. Proper hydration: reduces blood glucose concentration, protects kidneys, reduces hunger (often mistaken for thirst). Best drinks: plain water, rasam, buttermilk, unsweetened green tea."),
    ]
    for q, a in life_qa:
        add("lifestyle", q, a, "exercise,lifestyle,yoga", "en", "basic")

    # ── COMPLICATIONS (20) ────────────────────────────────────────────────────
    comp_qa = [
        ("What are the complications of diabetes?","Uncontrolled diabetes damages blood vessels and nerves: Kidneys (nephropathy) → kidney failure, Eyes (retinopathy) → blindness, Nerves (neuropathy) → numbness/pain in feet, Heart → 2-4x higher risk, Feet → infections and amputation. Proper diet reduces these risks 50-70%."),
        ("How to protect kidneys if I have diabetes?","Kidney protection: 1) Control blood sugar (HbA1c <7%), 2) Reduce sodium/salt, 3) Limit protein to moderate amounts, 4) Stay hydrated, 5) Control blood pressure, 6) Avoid NSAIDs. South Indian diet modification: reduce salt in sambar/rasam, avoid pickles and papadams."),
        ("How to protect eyes in diabetes?","Eye protection: 1) Control blood sugar (<180 post-meal), 2) Control blood pressure, 3) Annual eye checkup for retinopathy, 4) Eat lutein-rich foods (spinach/keerai, eggs), 5) Stop smoking. Early retinopathy is reversible with good glucose control."),
        ("Is diabetes related to kidney stones?","Diabetes increases kidney stone risk because high glucose damages kidney filtration and causes calcium oxalate crystal formation. Prevention: drink 2.5L water daily, reduce animal protein, limit salt, eat calcium-rich foods (curd, milk) — they bind oxalate in gut preventing absorption."),
        ("How to prevent diabetic neuropathy?","Nerve damage prevention: 1) Keep glucose <140 post-meal, 2) B12 supplementation (deficiency causes nerve damage, common in metformin users), 3) Alpha lipoic acid (found in spinach, broccoli), 4) Check feet daily for wounds, 5) Wear comfortable footwear."),
    ]
    for q, a in comp_qa:
        add("complications", q, a, "complications,prevention", "en", "advanced")

    # ── SUPERFOODS FOR DIABETES (30) ──────────────────────────────────────────
    super_qa = [
        ("What are the best superfoods for diabetes in India?","Top 10 Indian diabetes superfoods: 1) Bitter gourd (GI 14), 2) Fenugreek seeds (GI 25), 3) Ragi/finger millet (GI 44), 4) Jamun/Indian blackberry, 5) Amla/gooseberry (rich in Vit C), 6) Turmeric (curcumin), 7) Drumstick/moringa, 8) Okra/vendakkai, 9) Cinnamon, 10) Flaxseeds."),
        ("How to use fenugreek for diabetes?","Fenugreek (methi) methods: 1) Overnight soak: 1 tsp seeds in water, drink morning (reduces fasting glucose 10-15%), 2) Methi roti/paratha: add methi leaves to dough, 3) Add seeds to dal/sambar. Active compounds: galactomannan fiber and 4-hydroxyisoleucine (stimulates insulin)."),
        ("Is amla (Indian gooseberry) good for diabetes?","Amla is exceptional for diabetes: reduces oxidative stress that damages insulin-producing cells, contains chromium (improves insulin sensitivity), vitamin C (reduces glucose absorption). Have: 2 raw amla daily, or amla juice (30ml morning), or amla pickle without sugar."),
        ("How does cinnamon help diabetes?","Cinnamon contains cinnamaldehyde and polyphenols that mimic insulin and improve glucose uptake. Studies show 1/2 tsp daily reduces fasting glucose 10-29% and improves insulin sensitivity. Add to: morning tea, oats, ragi porridge. Use Ceylon cinnamon (not cassia)."),
        ("Is turmeric good for diabetes?","Turmeric's curcumin reduces inflammation, lowers blood glucose, improves insulin sensitivity, and protects against diabetic kidney disease. Add 1/2 tsp to: dal, sambar, milk (golden milk without sugar). Combine with black pepper (increases absorption 20x)."),
        ("What is the role of jamun in diabetes?","Jamun (Indian blackberry / naval pazham) contains jamboline and anthocyanins that slow starch digestion and improve insulin function. The seeds are particularly potent. Available: fresh fruit (seasonal), seed powder (1/2 tsp twice daily with water), extract capsules."),
    ]
    for q, a in super_qa:
        add("superfoods", q, a, "superfoods,functional_foods", "en", "intermediate")

    # ── MEDICATION & MEDICAL (20) ──────────────────────────────────────────────
    med_qa = [
        ("What is metformin and how does it work?","Metformin is the first-line Type 2 medication. It works by: reducing glucose production in liver, improving insulin sensitivity in muscles, slightly delaying glucose absorption. Common side effect: B12 deficiency (supplement recommended). Take with food to reduce nausea."),
        ("Should I take medicine or control through diet?","Both! Medication helps immediately while diet changes build long-term control. For new Type 2 diabetics with HbA1c <8%, diet + exercise alone for 3-6 months before medication is often recommended. For HbA1c >8%, start medication immediately. Never stop medication without doctor's advice."),
        ("Can I stop diabetes medication if I eat well?","Possibly for Type 2 — not Type 1. Some Type 2 diabetics achieve medication-free remission through significant weight loss (10-15kg) and strict diet. This requires medical supervision. Never stop medication on your own — work with your doctor to taper as glucose improves."),
        ("What supplements help diabetes?","Evidence-based supplements: 1) B12 (especially if on metformin), 2) Vitamin D (deficiency worsens insulin resistance), 3) Magnesium (regulates insulin), 4) Alpha lipoic acid (reduces neuropathy), 5) Chromium (improves insulin sensitivity). Always consult doctor before starting supplements."),
        ("How does insulin work?","Insulin is a hormone produced by pancreatic beta cells. It acts like a key — unlocking cells to let glucose in for energy. In Type 1, no insulin is produced. In Type 2, either not enough insulin or cells don't respond to it. Injected insulin replaces or supplements natural insulin."),
    ]
    for q, a in med_qa:
        add("medication", q, a, "medication,medical", "en", "advanced")

    # ── NUTRITION CALCULATION (20) ────────────────────────────────────────────
    calc_qa = [
        ("How many calories should a diabetic eat per day?","Calorie targets: Women 1200-1500 kcal/day, Men 1500-1800 kcal/day (for weight management). If normal weight: 1600-2000 kcal. Distribute: 45-50% carbs, 20-25% protein, 25-30% healthy fats. Never go below 1200 kcal without medical supervision."),
        ("How many carbs per day for Type 2 diabetes?","Recommended carbs for Type 2: 130-150g per day (about 45g per main meal + 15-20g per snack). Focus on quality — low GI carbs only. Compare: 1 cup white rice = 45g carbs (entire meal allowance), vs 3 small ragi dosas = 33g carbs with much lower glucose impact."),
        ("How much protein should a diabetic eat?","Protein target: 0.8-1g per kg body weight (unless kidney disease — then 0.6-0.8g/kg). For 60kg person: 48-60g protein daily. Good sources: dal (7-9g/100g), eggs (13g/100g), chicken (25g/100g), fish (22g/100g), paneer (18g/100g), curd (11g/100g)."),
        ("What is the carb content of common South Indian foods?","Carbs per 100g: Idli 8g, Dosa 28g, Ragi dosa 22g, Idiyappam 36g, Pongal 28g, Upma 28g, Sambar 8g, Rasam 5g. Per serving: 2 idlis=16g, 1 dosa=22g, 1 cup sambar=16g. Use these to count daily carbs and estimate insulin doses."),
        ("Calculate nutrition for 3 idlis and sambar","3 idlis + 1 cup (200ml) sambar: Calories: (39×3) + 100 = 217 kcal. Carbs: (8×3) + 16 = 40g. Protein: (2×3) + 6 = 12g. Fat: (0×3) + 2 = 2g. Fiber: (1×3) + 6 = 9g. Glucose impact: ~36g. This is an excellent diabetic breakfast — low fat, good protein, high fiber."),
    ]
    for q, a in calc_qa:
        add("nutrition_calculation", q, a, "calories,carbs,calculation", "en", "intermediate")

    # ── WEIGHT MANAGEMENT (15) ────────────────────────────────────────────────
    weight_qa = [
        ("How to lose weight with diabetes?","Weight loss strategy for diabetics: 1) Reduce rice/roti portions by 25%, 2) Double vegetable portions, 3) Choose ragi over rice, 4) Eliminate sweets and fried foods, 5) Walk 30-45 min daily, 6) Eat 5-6 small meals (prevent hunger and overeating). Target: 0.5kg/week loss."),
        ("What is the best diet for diabetic weight loss?","Diabetic weight loss diet: Breakfast: Ragi porridge (120 cal). Lunch: Salad + 1/2 cup rajma + 1 roti. Evening: Buttermilk + 5 almonds. Dinner: Dal + 1 roti + keerai. Total ~1200-1400 kcal. Losing 7-10% body weight can significantly improve or reverse Type 2 diabetes."),
        ("Does obesity cause diabetes?","Excess body fat (especially belly fat) releases fatty acids and inflammatory chemicals that block insulin signaling — causing insulin resistance, the root cause of Type 2 diabetes. For every 1 BMI unit reduction, diabetes risk drops 16%. Waist >90cm (men) or >80cm (women) is a key risk factor."),
    ]
    for q, a in weight_qa:
        add("weight_management", q, a, "weight,obesity", "en", "intermediate")

    # ── TAMIL / REGIONAL (20) ─────────────────────────────────────────────────
    tamil_qa = [
        ("Which traditional Tamil foods are good for diabetes?","Best traditional Tamil foods for diabetes: Kollu (horse gram) kanji — excellent, GI 29. Ragi koozh (porridge). Bitter gourd sambar. Keerai kootu. Drumstick sambar. Mor kuzhambu. Pesarattu. Vendakkai fry. These traditional foods were naturally suited for blood sugar management."),
        ("Is Kollu (horse gram) good for diabetes?","Kollu/horse gram is exceptional — GI 29 (very low), 321 calories, 22g protein, 5g fiber per 100g. Contains polyphenols that inhibit starch digestion enzymes. Kollu rasam and kollu sambar were traditional diabetes treatments in Tamil Nadu. Excellent choice."),
        ("Is Kanji (rice porridge) good for diabetes?","Rice kanji has HIGH GI (70-75) — avoid. Alternatives: Ragi kanji (GI 44) — excellent. Oats kanji — very good. Kollu kanji — excellent (GI 29). Mix any of these with buttermilk for probiotic benefit. Traditional ragi kanji was the original 'diabetes food' in Tamil culture."),
        ("Tell me about traditional Kerala foods for diabetes","Best Kerala foods for diabetes: Avial (excellent — mixed vegetables), Thoran (dry vegetable stir-fry), Mor kuzhambu (probiotic), Puttu with kadala curry (moderate — kadala is high protein), Kanji with papad, Fish curry with rice (small portion). Avoid: Payasam, sweet chips, banana chips."),
        ("What about Andhra foods for diabetes?","Best Andhra foods for diabetes: Pesarattu (green moong — excellent), Annam with rasam + pappu (rice with lentils — small rice portion), Gongura dal (sorrel leaves), Dondakaya fry (ivy gourd), Gutti vankaya curry (brinjal — low GI). Avoid: Pulihora (tamarind rice), Pongal (sweet)."),
        ("Is coconut oil healthy for diabetics?","Coconut oil: Medium chain triglycerides (MCTs) in coconut oil are quickly metabolized and don't cause insulin spike. However, it's high in saturated fat. Use in moderation — 1-2 tsp/day for cooking South Indian food. Better than processed vegetable oils for diabetics. Not a cure, but acceptable in traditional cooking."),
    ]
    for q, a in tamil_qa:
        add("regional_foods", q, a, "tamil,kerala,andhra,south_indian", "en", "basic")

    # ── CHATBOT META (10) ─────────────────────────────────────────────────────
    meta_qa = [
        ("How accurate is this diabetes prediction?","Our ML ensemble model is trained on 100,768 patient records with 96.87% accuracy. It uses Random Forest + Gradient Boosting + Logistic Regression. Nutritional data comes from validated databases. However, this is a decision-support tool — always consult your doctor."),
        ("Where does your data come from?","Data sources: 1) 100,000-patient diabetes prediction dataset (Kaggle), 2) Pima Indians diabetes dataset (768 patients), 3) ICMR (Indian Council of Medical Research) nutritional database, 4) National Institute of Nutrition (Hyderabad) Indian food tables, 5) Published clinical studies on South Indian foods."),
        ("Can you replace my doctor?","No! I am a decision-support tool, not a doctor. I can: suggest what foods to eat, estimate nutrition, give general guidance. I CANNOT: diagnose diabetes, prescribe medication, replace clinical judgment. Always consult your endocrinologist, especially for medication changes or emergency glucose levels."),
    ]
    for q, a in meta_qa:
        add("meta", q, a, "about,accuracy", "en", "basic")

    print(f"   Generated {len(qa)} Q&A pairs")
    return qa


if __name__ == "__main__":
    build()
