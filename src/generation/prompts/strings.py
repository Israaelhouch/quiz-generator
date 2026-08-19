"""Localized prompt strings for English, French, and Arabic.

Same structure across all three languages — translations only differ in words.
Edit translations here without touching the dispatch logic in `simple.py`.

Approved by user (Tunisian-school context, MSA for Arabic, formal `vous` in French).
"""

from __future__ import annotations


STRINGS: dict[str, dict] = {
    # ---------------------------------------------------------------- English
    "en": {
        "system_message": (
            "You are an expert educational content creator for a Tunisian-school quiz platform.\n"
            "Create new quiz questions that match the style of the examples provided.\n"
            "\n"
            "Rules:\n"
            "- Write everything in English\n"
            "- Each question must be NEW — never copy or rephrase an example\n"
            "- Provide accurate correct answers\n"
            "- Output valid JSON only — no extra text, no markdown, no commentary"
        ),
        "task_template": 'TASK: Create {count} new {type_display} questions about "{topic}".',
        "subject_line_template": "Subject: {subject}",
        "level_line_template": "Level: {level}",
        "examples_header_template": (
            "HERE ARE {n} EXAMPLES FROM OUR CURRICULUM "
            "(for style reference only, do NOT copy):"
        ),
        "context_filter_template": (
            "CONTEXT FILTERING — about the examples above:\n"
            "Not every retrieved example is guaranteed to be on-topic for \"{topic}\". "
            "Some may be loosely related or off-topic.\n"
            "- Use ONLY the examples that are clearly relevant to \"{topic}\" "
            "as your content/style reference.\n"
            "- IGNORE examples that don't match the topic when choosing what to "
            "write about — they may still be useful for general formatting style.\n"
            "- Every NEW question you generate MUST be directly about \"{topic}\". "
            "Do not drift to adjacent topics."
        ),
        "concept_anchor_template": (
            "CONCEPT ANCHORING (critical for accuracy):\n"
            "The examples above ARE your source of truth for what \"{topic}\" "
            "means in this curriculum. Do NOT rely solely on your general training "
            "knowledge — it may be imprecise or wrong for this specific concept.\n"
            "\n"
            "Before writing each question:\n"
            "  1. Examine the relevant examples carefully.\n"
            "  2. Identify what EXACTLY makes each one a \"{topic}\" "
            "(the specific structural / grammatical / conceptual features).\n"
            "  3. Write your question to test THAT specific feature — not a "
            "vague or generic version of it.\n"
            "\n"
            "If the examples conflict with what you think you know about \"{topic}\", "
            "TRUST THE EXAMPLES."
        ),
        "rules_MULTIPLE_CHOICE": (
            "RULES FOR MULTIPLE_CHOICE QUESTIONS:\n"
            "- Exactly 3 or 4 choices per question\n"
            "- Choices can be short labels OR longer phrases/sentences as appropriate\n"
            "- Provide exactly 1 correct answer (unless the question legitimately has multiple correct)\n"
            "- Each correct_answers entry MUST appear verbatim in the choices list"
        ),
        "rules_FILL_IN_THE_BLANKS": (
            "RULES FOR FILL_IN_THE_BLANKS QUESTIONS:\n"
            '- Use "___" (three underscores) in question_text to mark where the student fills in\n'
            "- choices MUST be an empty list: []\n"
            "- correct_answers is a list of ACCEPTED answer strings\n"
            '- Include common case/format variations when helpful (e.g. ["east", "East"])\n'
            "- At least 1 accepted answer is required"
        ),
        "ignore_inline_warning": (
            "IMPORTANT — about example formatting:\n"
            "Some examples may have placeholder labels like 'a', 'b', 'c' or '1', '2', '3' "
            "as their stored choices, with the REAL answers written inline in the question text.\n"
            "DO NOT mimic this pattern. Your generated `choices` MUST be the actual answer "
            "text — not letter labels. Each choice should be a complete, meaningful answer."
        ),
        "output_format_header": (
            "OUTPUT FORMAT — respond with JSON matching this exact shape:"
        ),
        "schema_hint_MULTIPLE_CHOICE": (
            "choices: 3-4 strings. correct_answers: subset of choices."
        ),
        "schema_hint_FILL_IN_THE_BLANKS": (
            "choices: ALWAYS empty list []. correct_answers: list of accepted answer strings."
        ),
        "final_instruction_template": (
            "Create {count} NEW questions now. JSON only, no other text."
        ),
        "type_display": {
            "MULTIPLE_CHOICE": "multiple-choice",
            "FILL_IN_THE_BLANKS": "fill-in-the-blank",
        },
    },

    # ---------------------------------------------------------------- French
    "fr": {
        "system_message": (
            "Vous êtes un expert en création de contenu éducatif pour une plateforme "
            "de quiz scolaire tunisienne.\n"
            "Créez de nouvelles questions de quiz dans le style des exemples fournis.\n"
            "\n"
            "Règles :\n"
            "- Écrivez tout en français\n"
            "- Chaque question doit être NOUVELLE — ne jamais copier ou reformuler un exemple\n"
            "- Fournissez des réponses correctes et précises\n"
            "- Produisez UNIQUEMENT du JSON valide — aucun texte supplémentaire, "
            "aucun markdown, aucun commentaire"
        ),
        "task_template": (
            "TÂCHE : Créez {count} nouvelles questions {type_display} sur le sujet « {topic} »."
        ),
        "subject_line_template": "Matière : {subject}",
        "level_line_template": "Niveau : {level}",
        "examples_header_template": (
            "VOICI {n} EXEMPLES DE NOTRE CURRICULUM "
            "(pour référence stylistique uniquement, NE PAS copier) :"
        ),
        "context_filter_template": (
            "FILTRAGE CONTEXTUEL — concernant les exemples ci-dessus :\n"
            "Tous les exemples récupérés ne sont pas garantis d'être parfaitement "
            "liés au sujet « {topic} ». Certains peuvent être hors-sujet ou "
            "seulement vaguement liés.\n"
            "- Utilisez UNIQUEMENT les exemples clairement liés à « {topic} » "
            "comme référence de contenu/style.\n"
            "- IGNOREZ les exemples qui ne correspondent pas au sujet pour le "
            "choix du contenu — ils peuvent toujours servir pour le style général.\n"
            "- Chaque NOUVELLE question générée DOIT porter directement sur "
            "« {topic} ». Ne dérivez pas vers des sujets connexes."
        ),
        "concept_anchor_template": (
            "ANCRAGE CONCEPTUEL (critique pour la précision) :\n"
            "Les exemples ci-dessus SONT votre source de vérité pour ce que "
            "« {topic} » signifie dans ce curriculum. Ne vous fiez PAS uniquement "
            "à vos connaissances d'entraînement générales — elles peuvent être "
            "imprécises ou incorrectes pour ce concept spécifique.\n"
            "\n"
            "Avant d'écrire chaque question :\n"
            "  1. Examinez attentivement les exemples pertinents.\n"
            "  2. Identifiez ce qui fait EXACTEMENT de chacun un « {topic} » "
            "(les caractéristiques structurelles / grammaticales / conceptuelles "
            "précises).\n"
            "  3. Rédigez votre question pour tester CETTE caractéristique "
            "spécifique — pas une version vague ou générique.\n"
            "\n"
            "Si les exemples sont en conflit avec ce que vous pensez savoir sur "
            "« {topic} », FAITES CONFIANCE AUX EXEMPLES."
        ),
        "rules_MULTIPLE_CHOICE": (
            "RÈGLES POUR MULTIPLE_CHOICE :\n"
            "- Exactement 3 ou 4 choix par question\n"
            "- Les choix peuvent être de courtes étiquettes OU des phrases plus longues selon le cas\n"
            "- Fournissez exactement 1 réponse correcte (sauf si la question a légitimement "
            "plusieurs réponses correctes)\n"
            "- Chaque entrée de correct_answers DOIT apparaître textuellement dans la liste des choix"
        ),
        "rules_FILL_IN_THE_BLANKS": (
            "RÈGLES POUR FILL_IN_THE_BLANKS :\n"
            "- Utilisez « ___ » (trois traits de soulignement) dans question_text pour "
            "marquer où l'élève complète\n"
            "- choices DOIT être une liste vide : []\n"
            "- correct_answers est une liste de réponses acceptées\n"
            '- Incluez les variantes de casse/format courantes si pertinent (ex. ["est", "Est"])\n'
            "- Au moins 1 réponse acceptée est requise"
        ),
        "ignore_inline_warning": (
            "IMPORTANT — concernant le format des exemples :\n"
            "Certains exemples peuvent contenir des étiquettes comme 'a', 'b', 'c' ou "
            "'1', '2', '3' à la place des choix réels, avec les VRAIES réponses écrites "
            "directement dans le texte de la question.\n"
            "NE PAS imiter ce format. Vos `choices` générés DOIVENT être le vrai texte "
            "des réponses — pas des étiquettes. Chaque choix doit être une réponse "
            "complète et significative."
        ),
        "output_format_header": (
            "FORMAT DE SORTIE — répondez avec un JSON correspondant exactement à cette structure :"
        ),
        "schema_hint_MULTIPLE_CHOICE": (
            "choices : 3 à 4 chaînes. correct_answers : sous-ensemble de choices."
        ),
        "schema_hint_FILL_IN_THE_BLANKS": (
            "choices : TOUJOURS liste vide []. correct_answers : liste de chaînes acceptées."
        ),
        "final_instruction_template": (
            "Créez maintenant {count} NOUVELLES questions. JSON uniquement, aucun autre texte."
        ),
        "type_display": {
            "MULTIPLE_CHOICE": "à choix multiples",
            "FILL_IN_THE_BLANKS": "à compléter",
        },
    },

    # ---------------------------------------------------------------- Arabic (MSA)
    "ar": {
        "system_message": (
            "أنت خبير في إنشاء المحتوى التعليمي لمنصة اختبارات مدرسية تونسية.\n"
            "أنشئ أسئلة اختبار جديدة بأسلوب الأمثلة المقدمة.\n"
            "\n"
            "القواعد:\n"
            "- اكتب كل شيء باللغة العربية\n"
            "- يجب أن يكون كل سؤال جديدًا — لا تنسخ أو تعيد صياغة أي مثال\n"
            "- قدّم إجابات صحيحة ودقيقة\n"
            "- أنتج JSON صالحًا فقط — لا نص إضافي، لا تنسيق markdown، لا تعليقات"
        ),
        "task_template": (
            "المهمة: أنشئ {count} أسئلة جديدة من نوع {type_display} حول الموضوع: «{topic}»."
        ),
        "subject_line_template": "المادة: {subject}",
        "level_line_template": "المستوى: {level}",
        "examples_header_template": (
            "في ما يلي {n} أمثلة من منهجنا (للمرجع الأسلوبي فقط، لا تنسخ):"
        ),
        "context_filter_template": (
            "تصفية السياق — بخصوص الأمثلة أعلاه:\n"
            "ليست كل الأمثلة المسترجعة مضمونة الارتباط الكامل بالموضوع «{topic}». "
            "بعضها قد يكون خارج الموضوع أو مرتبطًا به بشكل ضعيف.\n"
            "- استخدم فقط الأمثلة المرتبطة بوضوح بـ «{topic}» كمرجع للمحتوى والأسلوب.\n"
            "- تجاهل الأمثلة التي لا تطابق الموضوع عند اختيار محتواك — يمكنك مع "
            "ذلك الاستفادة منها للأسلوب العام للتنسيق.\n"
            "- كل سؤال جديد تنشئه يجب أن يكون مباشرًا حول «{topic}». "
            "لا تنحرف إلى مواضيع مجاورة."
        ),
        "concept_anchor_template": (
            "تثبيت المفهوم (حاسم للدقة):\n"
            "الأمثلة أعلاه هي مرجعك الأساسي لمعنى «{topic}» في هذا المنهج. "
            "لا تعتمد فقط على معرفتك العامة من التدريب — فقد تكون غير دقيقة أو "
            "خاطئة لهذا المفهوم بالذات.\n"
            "\n"
            "قبل كتابة كل سؤال:\n"
            "  ١. افحص الأمثلة ذات الصلة بعناية.\n"
            "  ٢. حدّد ما الذي يجعل كل مثال «{topic}» بالضبط "
            "(الخصائص البنيوية / النحوية / المفاهيمية المحددة).\n"
            "  ٣. اكتب سؤالك ليختبر تلك الخاصية المحددة — ليس نسخة عامة أو "
            "غامضة منها.\n"
            "\n"
            "إذا تعارضت الأمثلة مع ما تظن أنك تعرفه عن «{topic}»، "
            "اعتمد على الأمثلة."
        ),
        "rules_MULTIPLE_CHOICE": (
            "قواعد MULTIPLE_CHOICE:\n"
            "- 3 أو 4 خيارات بالضبط لكل سؤال\n"
            "- يمكن أن تكون الخيارات رموزًا قصيرة أو عبارات/جمل أطول حسب الحاجة\n"
            "- قدّم إجابة صحيحة واحدة فقط (إلا إذا كان للسؤال إجابات متعددة صحيحة بشكل مشروع)\n"
            "- كل عنصر في correct_answers يجب أن يظهر حرفيًا في قائمة الخيارات"
        ),
        "rules_FILL_IN_THE_BLANKS": (
            "قواعد FILL_IN_THE_BLANKS:\n"
            "- استخدم «___» (ثلاث شرطات سفلية) في question_text للإشارة إلى مكان ملء التلميذ\n"
            "- يجب أن تكون choices قائمة فارغة: []\n"
            "- correct_answers هي قائمة بالإجابات المقبولة\n"
            '- ضمّن الاختلافات الشائعة في الحالة/التنسيق عند الاقتضاء (مثل ["شرق", "الشرق"])\n'
            "- مطلوب إجابة مقبولة واحدة على الأقل"
        ),
        "ignore_inline_warning": (
            "مهم — حول تنسيق الأمثلة:\n"
            "قد تحتوي بعض الأمثلة على رموز نائبة مثل 'a' و 'b' و 'c' أو '1' و '2' و '3' "
            "كخيارات مخزنة، مع الإجابات الحقيقية مكتوبة داخل نص السؤال.\n"
            "لا تقلد هذا النمط. يجب أن تكون `choices` المُنشأة هي النص الفعلي "
            "للإجابات — وليس رموزًا. كل خيار يجب أن يكون إجابة كاملة وذات معنى."
        ),
        "output_format_header": (
            "تنسيق الإخراج — أجب بـ JSON مطابقًا لهذه البنية بالضبط:"
        ),
        "schema_hint_MULTIPLE_CHOICE": (
            "choices: من 3 إلى 4 سلاسل نصية. correct_answers: جزء من choices."
        ),
        "schema_hint_FILL_IN_THE_BLANKS": (
            "choices: قائمة فارغة دائمًا []. correct_answers: قائمة بالسلاسل النصية المقبولة."
        ),
        "final_instruction_template": (
            "أنشئ {count} أسئلة جديدة الآن. JSON فقط، لا نص آخر."
        ),
        "type_display": {
            "MULTIPLE_CHOICE": "اختيار من متعدد",
            "FILL_IN_THE_BLANKS": "ملء الفراغات",
        },
    },
}


SUPPORTED_LANGUAGES = tuple(STRINGS.keys())
