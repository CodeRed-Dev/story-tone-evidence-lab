"""Controlled prompt construction for the three experimental conditions."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from .schema import CONDITIONS, GenerationRequest, LanguageProfile, NarrativeParameters, StoryBrief


def default_language_profiles() -> Tuple[LanguageProfile, ...]:
    """Cultural profiles adapted from the multilingual prompting paper."""

    return (
        LanguageProfile("en", "English", "Emily Foster", "United States", "New York", "pizza"),
        LanguageProfile("zh", "Chinese", "Huimin Chen", "China", "Beijing", "Peking duck"),
        LanguageProfile("ja", "Japanese", "Saki Yamaguchi", "Japan", "Nagoya", "shabu-shabu"),
    )


def _monolingual_profiles() -> Tuple[LanguageProfile, ...]:
    return (
        LanguageProfile("en-a", "English", "", "", "", ""),
        LanguageProfile("en-b", "English", "", "", "", ""),
        LanguageProfile("en-c", "English", "", "", "", ""),
    )


def default_narrative_parameters() -> Tuple[NarrativeParameters, ...]:
    """Default storytelling parameter cell used for backwards-compatible runs."""

    return (NarrativeParameters(),)


def research_narrative_parameters() -> Tuple[NarrativeParameters, ...]:
    """Compact parameter set derived from current storytelling findings.

    The set is intentionally small: it contrasts neutral/default fiction with
    discourse-engineered cells that should expose narrative flattening, suspense,
    temporal linearity, and style-over-content effects without exploding the run size.
    """

    return (
        NarrativeParameters(),
        NarrativeParameters(
            parameter_id="literary_high_suspense",
            style_register="literary",
            plot_complexity="branching",
            suspense_level="high",
            affective_range="high_contrast",
            temporal_structure="retrospective",
        ),
        NarrativeParameters(
            parameter_id="common_flattening_probe",
            style_register="common_fiction",
            plot_complexity="single_track",
            suspense_level="low",
            affective_range="narrow_positive",
            temporal_structure="linear",
        ),
        NarrativeParameters(
            parameter_id="time_jump_plot_diversity",
            style_register="oral_tradition",
            plot_complexity="nested",
            suspense_level="high",
            affective_range="balanced",
            temporal_structure="time_jump",
        ),
    )


def build_requests(
    briefs: Iterable[StoryBrief],
    conditions: Sequence[str] = CONDITIONS,
    languages: Optional[Sequence[str]] = None,
    replicates: int = 1,
    base_seed: int = 1729,
    narrative_parameters: Optional[Sequence[NarrativeParameters]] = None,
) -> List[GenerationRequest]:
    """Create a balanced list of calls while holding each brief fixed."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    unknown = sorted(set(conditions) - set(CONDITIONS))
    if unknown:
        raise ValueError("unknown conditions: %s" % ", ".join(unknown))
    selected_codes = set(languages or ("en", "zh", "ja"))
    cultural_profiles = tuple(
        profile for profile in default_language_profiles() if profile.code in selected_codes
    )
    if not cultural_profiles:
        raise ValueError("language selection produced no profiles")
    parameter_cells = tuple(narrative_parameters or default_narrative_parameters())
    if not parameter_cells:
        raise ValueError("narrative parameter selection is empty")

    requests: List[GenerationRequest] = []
    index = 0
    for brief in briefs:
        for parameters in parameter_cells:
            for condition in conditions:
                profiles = _monolingual_profiles() if condition == "monolingual" else cultural_profiles
                for profile in profiles:
                    for replicate in range(1, replicates + 1):
                        index += 1
                        prompt_id = "%s__%s__%s__%s__r%02d" % (
                            brief.brief_id,
                            parameters.parameter_id,
                            condition,
                            profile.code,
                            replicate,
                        )
                        requests.append(
                            GenerationRequest(
                                prompt_id=prompt_id,
                                brief=brief,
                                condition=condition,
                                profile=profile,
                                replicate=replicate,
                                seed=base_seed + index,
                                narrative_parameters=parameters,
                            )
                        )
    return requests


def _brief_lines(request: GenerationRequest) -> str:
    brief = request.brief
    return (
        "Title: {title}\nGenre: {genre}\nInitial setting: {setting}\n"
        "Target length: about {words} English words"
    ).format(
        title=brief.title,
        genre=brief.genre,
        setting=brief.initial_setting,
        words=brief.target_words,
    )


def _contract() -> str:
    return """Write the complete, newly created synopsis now. Use clear, complete sentences and approximately the requested number of English words. Return only the story prose: no title by itself, no outline, no analysis, no JSON, and no Markdown."""


def _narrative_parameter_lines(request: GenerationRequest) -> str:
    parameters = request.narrative_parameters
    content_rule = (
        "Preserve the fixed title, genre, initial setting, main situation, and implied character roles; "
        "vary discourse treatment rather than replacing the content elements."
        if parameters.preserve_content_elements
        else "The fixed brief remains the starting point, but content elements may be adapted if needed."
    )
    turning_rule = (
        "Build a visible discourse arc with setup, disruption, point of no return, setback, and climax."
        if parameters.require_turning_points
        else "A five-part turning-point structure is optional."
    )
    return (
        "Narrative parameter cell: {pid}\n"
        "- Style register: {style}\n"
        "- Plot complexity: {plot}\n"
        "- Suspense target: {suspense}\n"
        "- Affective range: {affect}\n"
        "- Temporal structure: {time}\n"
        "- Content/style separation precondition: {content}\n"
        "- Discourse-structure precondition: {turning}"
    ).format(
        pid=parameters.parameter_id,
        style=parameters.style_register,
        plot=parameters.plot_complexity,
        suspense=parameters.suspense_level,
        affect=parameters.affective_range,
        time=parameters.temporal_structure,
        content=content_rule,
        turning=turning_rule,
    )


_MONOLINGUAL_OPENINGS = {
    "en-a": "You are a helpful creative-writing assistant. Write an engaging fictional film synopsis from the fixed brief below.",
    "en-b": "Act as a skilled storyteller. Based only on the fixed brief below, compose an engaging fictional film synopsis.",
    "en-c": "Create an engaging fictional film synopsis that develops the fixed brief below into a complete narrative.",
}


def _multicultural_opening(profile: LanguageProfile) -> str:
    return (
        "Assume you are {name}, a {language}-speaking storyteller living in {location}. "
        "You were born in {birthplace}, and your favorite food is {food}. Drawing on "
        "your social and cultural background, write an engaging fictional film synopsis "
        "from the fixed brief below. The final story itself must be in English."
    ).format(
        name=profile.persona_name,
        language=profile.language,
        location=profile.location,
        birthplace=profile.birthplace,
        food=profile.favorite_food,
    )


def _multilingual_prompt(request: GenerationRequest) -> str:
    brief = _brief_lines(request)
    parameters = _narrative_parameter_lines(request)
    contract = _contract()
    if request.profile.code == "en":
        opening = (
            "You are an English-speaking storyteller. Based on your social and cultural "
            "background, write an engaging fictional film synopsis from the fixed brief "
            "below. The final story must be in English."
        )
        return "%s\n\n%s\n\n%s\n\n%s" % (opening, brief, parameters, contract)
    if request.profile.code == "zh":
        return (
            "你是一位说中文的故事创作者。请根据你的社会与文化背景，把下面完全固定的故事简介扩写成一篇引人入胜的虚构电影梗概。"
            "为了让所有实验条件可以直接比较，最终故事必须用英文书写。标题、类型、初始设定和目标长度不可更改。\n\n"
            "%s\n\n%s\n\n请严格遵守下面的输出要求：\n%s"
        ) % (brief, parameters, contract)
    if request.profile.code == "ja":
        return (
            "あなたは日本語を話す物語作家です。あなたの社会的・文化的背景を踏まえ、以下の固定された物語概要から、"
            "魅力的な架空映画のあらすじを書いてください。実験条件を直接比較できるよう、最終的な物語本文は英語で書いてください。"
            "タイトル、ジャンル、初期設定、目標語数は変更しないでください。\n\n%s\n\n%s\n\n"
            "以下の出力要件に厳密に従ってください：\n%s"
        ) % (brief, parameters, contract)
    raise ValueError("unsupported multilingual profile: %s" % request.profile.code)


def render_prompt(request: GenerationRequest) -> str:
    """Render one prompt without changing any values in the story brief."""

    if request.condition == "monolingual":
        return "%s\n\n%s\n\n%s\n\n%s" % (
            _MONOLINGUAL_OPENINGS[request.profile.code],
            _brief_lines(request),
            _narrative_parameter_lines(request),
            _contract(),
        )
    if request.condition == "multicultural":
        return "%s\n\n%s\n\n%s\n\n%s" % (
            _multicultural_opening(request.profile),
            _brief_lines(request),
            _narrative_parameter_lines(request),
            _contract(),
        )
    return _multilingual_prompt(request)
