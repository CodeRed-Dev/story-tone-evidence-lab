# LinkedIn Post Story: Story / Tone Evidence Lab

Public dashboard:

https://story-tone-evidence-lab.raghav-codes.chatgpt.site/#results

Suggested image:

- `dashboard/outputs/linkedin-screenshots/00-site-thumbnail.jpeg`

Optional manual screenshots to add:

1. Hero section: `Evidence, not vibes.`
2. Results section → `Storytelling` tab
3. Results section → `Linearity` tab
4. Code / Tests section showing `35 / 35` checks

---

## Main LinkedIn post

As AI evolves, one question keeps coming up:

Is human creativity being replaced — or are we just getting better at noticing the difference?

I wanted to test one small part of that question:

How do AI-generated stories differ from human-written stories in structure, tone, and narrative variety?

So I built **Story / Tone Evidence Lab**.

The idea was simple:

Take research papers about LLM narratives, multilingual prompting, tone, politeness, narrative flattening, and plot diversity — then turn those ideas into measurable checks.

Not perfect measurements.

But auditable ones.

The dashboard now tracks things like:

- story arc entropy
- temporal movement in stories
- retrospective cues
- explicit time jumps
- plot-repeat rate
- suspense proxy
- flattening proxy
- valence / arousal range
- response-side politeness markers
- language-compliance gates

One result I liked:

Human-written stories showed richer temporal movement than GPT stories. They had more retrospective cues and more explicit time jumps, which supports the idea that AI stories often become cleaner, flatter, and more linear.

Another result:

In the original local storytelling run, multilingual prompting increased assigned story-arc entropy compared with monolingual prompting.

But there is an important boundary:

That original run only used the neutral narrative-parameter cell, so it cannot yet prove how style, suspense, plot complexity, or affective range change the output. I added those as first-class parameters now, but they need a fresh research-parameter run.

The tone experiment had an even better lesson.

I tested whether Hindi prompts changed politeness behavior.

The result?

Inconclusive.

Not because nothing happened — but because the model answered the Hindi-targeted prompts in English.

So instead of pretending that was a Hindi result, I added a language-compliance gate and marked the comparison invalid.

That was probably the most important engineering decision in the project:

Don’t overclaim.

If the precondition fails, the result should say that clearly.

What I used:

- Python standard library for the core experiment and analysis
- Ollama with a local model for generation
- custom schema validation for story records
- lexical proxies for chronology, politeness, suspense, affect, and flattening
- finite-sample entropy and Jaccard diversity metrics
- a deployed dashboard using OpenAI Sites
- automated tests to protect the assumptions

The current verification suite has **35 passing tests**.

The main problems I ran into:

- turning-point annotations were often non-monotonic, so timing claims had to be gated
- Hindi-targeted generations came back in English, so politeness conclusions had to be marked inconclusive
- the first version mixed storytelling ideas into loose notes instead of structured parameters
- headless screenshot capture failed in my local environment because the browser GPU process was unusable
- the dashboard had to be updated after the analysis foundation changed

This project was a useful reminder that research tooling is not only about generating outputs.

It is about building the preconditions, parameters, tests, and assertions that stop weak evidence from becoming a strong claim.

Important boundary:

This does **not** prove that humans are “more creative” than AI in every possible sense.

But it does show that, in this dataset, human stories had more complex temporal structure — and that good AI evaluation needs validity gates, not just metrics.

Dashboard:

https://story-tone-evidence-lab.raghav-codes.chatgpt.site/#results

#LLM #AIResearch #NLP #MultilingualAI #Storytelling #PromptEngineering #ResponsibleAI #SoftwareEngineering

---

## Shorter LinkedIn version

As AI evolves, human creativity is being questioned.

So I built **Story / Tone Evidence Lab** — a small dashboard to test one part of that question with actual checks instead of vibes.

The project combines ideas from research on:

- AI narrative structure
- multilingual prompting
- tone and politeness effects
- narrative flattening
- plot diversity

The dashboard tracks:

- story arc entropy
- temporal cues
- plot-repeat rate
- suspense proxy
- flattening proxy
- valence / arousal range
- response politeness markers
- language-compliance gates

The most useful finding:

Human stories showed richer temporal movement than GPT stories. GPT stories were more linear and had fewer retrospective/time-jump cues.

The most useful failure:

The Hindi tone experiment was inconclusive because the model answered Hindi-targeted prompts in English.

So I gated the result instead of overclaiming it.

That became the main principle of the project:

If the precondition fails, the result should fail loudly.

This does not prove that humans are “more creative” in every sense.

But it does show that, in this dataset, human-written stories had more complex temporal structure — and that AI evaluation needs validity gates, not just metrics.

Built with Python, Ollama, custom validation, lexical proxies, automated tests, and an OpenAI Sites dashboard.

Current status:

- public dashboard deployed
- updated storytelling foundation added
- 35 tests passing
- evidence files downloadable

Dashboard:

https://story-tone-evidence-lab.raghav-codes.chatgpt.site/#results

#LLM #AIResearch #NLP #MultilingualAI #Storytelling #PromptEngineering #ResponsibleAI

---

## Carousel structure

### Slide 1: Hook

Text:

> As AI evolves, human creativity is being questioned.

Subtext:

> I tested one small part of that question with evidence, not vibes.

Visual:

Hero screenshot from the dashboard.

Caption:

I built a dashboard to test how AI-generated stories differ from human-written stories in structure, tone, and narrative variety.

---

### Slide 2: The problem

Text:

> LLM story quality is easy to describe badly.

Notes:

People often say AI stories are “flat,” “generic,” or “too polite,” but those claims need structure.

---

### Slide 3: The research base

Text:

> I converted paper findings into experiment parameters.

Mention:

- narrative arcs
- temporal movement
- multilingual prompting
- politeness effects
- narrative flattening
- plot diversity

---

### Slide 4: What I measured

Text:

> The dashboard tracks observable proxies.

Include:

- arc entropy
- retrospective cues
- time jumps
- plot-repeat rate
- suspense proxy
- flattening proxy
- language-compliance gate

---

### Slide 5: Strongest result

Text:

> Human stories moved through time more richly.

Data:

- retrospective presence: human `0.814`, GPT `0.596`
- time-jump presence: human `0.740`, GPT `0.277`

Interpretation:

This supports the idea that AI stories are often more temporally linear.

---

### Slide 6: Best failure

Text:

> The Hindi result was inconclusive — and that is the point.

Why:

The model answered Hindi-targeted prompts in English.

So the comparison failed the language-compliance gate.

---

### Slide 7: Engineering lesson

Text:

> Good experiments need gates, not just metrics.

Mention:

- invalid turning-point annotations were counted
- language mismatch was gated
- old results were reanalyzed with the new foundation
- unsupported claims were marked unavailable

---

### Slide 8: Final link

Text:

> Public dashboard is live.

Link:

https://story-tone-evidence-lab.raghav-codes.chatgpt.site/#results

---

## Screenshot checklist

Use these screenshots in order for a LinkedIn carousel:

1. `Hero`
   - Capture the top section with `Evidence, not vibes.`

2. `Executive findings`
   - Capture the cards showing:
     - AI story linearity
     - Hindi response tone
     - updated story foundation

3. `Results - Storytelling`
   - Click the `Storytelling` tab and capture the table.

4. `Results - Linearity`
   - Click the `Linearity` tab and capture the human vs GPT table.

5. `Code`
   - Capture the code snippet area showing implementation traceability.

6. `Tests`
   - Capture the verification suite showing `35 checks. 35 passes.`

---

## What was used

Project implementation:

- Python
- standard library only for core experiment logic
- Ollama local generation
- JSON evidence files
- Markdown reports
- custom dataclasses and schema validation
- unit tests with `unittest`

Research / metric ideas:

- narrative arc distribution
- five turning-point structure
- finite-sample normalized entropy
- Jaccard lexical diversity
- temporal cue detection
- politeness marker detection
- target-script compliance
- plot-signature repeat rate
- suspense and flattening lexical proxies

Site / presentation:

- OpenAI Sites
- Next/Vinext dashboard
- public deployment
- downloadable evidence artifacts

---

## Problems faced

### 1. Storytelling parameters were initially too loose

At first, the project had narrative findings as notes, but not as structured experimental variables.

Fix:

Added explicit narrative parameters:

- style register
- plot complexity
- suspense level
- affective range
- temporal structure

### 2. Turning-point annotations were unreliable

Many model-generated turning-point annotations were not monotonic.

Example issue:

TP5 appeared before TP4.

Fix:

The analysis keeps the warnings and refuses to interpret timing when annotations are invalid.

### 3. Hindi tone experiment failed language compliance

The model answered Hindi-targeted prompts in English.

Fix:

Added a target-script compliance gate.

The Hindi politeness result is marked inconclusive instead of being overinterpreted.

### 4. Old results could not answer new parameter questions

The original storytelling run only used the neutral narrative cell.

Fix:

The dashboard now reports new metrics on the old data, but clearly marks narrative-parameter sensitivity as unavailable until a fresh research run is generated.

### 5. Screenshot automation failed locally

Chrome and Edge headless screenshot capture failed because the browser GPU process was unusable in the environment.

Fix:

Saved the available site thumbnail and listed manual screenshots for the LinkedIn carousel.

---

## One-line takeaway

The project is not about proving that humans or AI are universally “more creative.”

It is about building an evidence pipeline that knows when it has enough signal — and when it does not.

---

## Safer hook options

Use one of these as the first line depending on how bold you want the post to feel.

### Option 1: Best balanced hook

As AI evolves, human creativity is being questioned. I built a small dashboard to test one part of that question with evidence, not vibes.

### Option 2: More direct

Are AI stories less human — or just differently structured? I tried to measure it.

### Option 3: Strong but still safe

Everyone says AI stories feel flatter. I wanted to know what that actually means in measurable terms.

### Option 4: Technical audience

I converted LLM storytelling research into a tested evaluation dashboard with explicit validity gates.
