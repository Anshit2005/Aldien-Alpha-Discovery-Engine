import pytest

from python.dsl.parser import parse
from python.dsl.simplify import simplify
from python.dsl.validator import ValidationError, validate
from python.evaluation.fitness import FitnessResult
from python.search.evolution import EvolutionConfig, EvolutionSearch, train_score
from python.search.random_search import RandomSearch, RandomSearchConfig
from tests.search.test_evolution import make_research_data


def _search(seed=3, max_depth=6, **overrides):
    config = EvolutionConfig(
        population_size=8,
        generations=0,
        elite_count=2,
        tournament_size=2,
        max_depth=max_depth,
        seed=seed,
        validation_top_k=0,
        **overrides,
    )
    return EvolutionSearch(make_research_data(), config)


def _parents(search, count=40):
    trees = [search._validated(expression) for expression in EvolutionSearch.SEEDS]
    trees += [search._random_tree(search.config.max_depth) for _ in range(count)]
    return trees


def _fitness(icir, coverage):
    return FitnessResult(
        horizon=20, mean_ic=0.01, ic_std=0.1, icir=icir,
        num_days=10, num_observations=100, coverage=coverage,
    )


def test_random_trees_respect_depth_and_validate():

    search = _search(max_depth=5)
    for _ in range(500):
        tree = search._random_tree(5)
        assert EvolutionSearch._depth(tree) <= 5
        validate(tree)


def test_paths_subtree_and_replace_round_trip():

    tree = parse("DIV(SUB(CLOSE,TS_MEAN(CLOSE,20)),TS_STD(CLOSE,20))")
    paths = EvolutionSearch._paths(tree)

    # One path per node: DIV, SUB, CLOSE, TS_MEAN, CLOSE, TS_STD, CLOSE.
    assert len(paths) == 7
    for path in paths:
        subtree = EvolutionSearch._subtree(tree, path)
        assert EvolutionSearch._replace(tree, path, subtree) == tree

    replaced = EvolutionSearch._replace(tree, ("left", "right"), parse("OPEN"))
    assert str(replaced) == "DIV(SUB(CLOSE,OPEN),TS_STD(CLOSE,20))"


def test_mutation_changes_trees_and_keeps_them_well_formed():

    search = _search()
    changed = 0
    for parent in _parents(search):
        for _ in range(5):
            child = search._mutate(parent)
            # Mutation may exceed the depth limit; _next_generation rejects
            # those, so only shape and operator validity are required here.
            try:
                validate(child)
            except ValidationError as error:
                assert "depth" in str(error).lower()
            changed += child != parent
    assert changed > 0.8 * 5 * len(_parents(search))


def test_crossover_takes_a_subtree_from_the_other_parent():

    search = _search()
    left = parse("SUB(CLOSE,OPEN)")
    right = parse("TS_MAX(VOLUME,5)")
    right_parts = {str(EvolutionSearch._subtree(right, path)) for path in EvolutionSearch._paths(right)}

    for _ in range(50):
        child = search._crossover(left, right)
        validate(child)
        assert any(part in str(child) for part in right_parts)


def test_next_generation_is_unique_canonical_and_within_depth():

    search = _search(max_depth=5)
    scored = search._score_population(search._initial_population())

    population = search._next_generation(scored)

    expressions = [str(node) for node in population]
    assert len(population) == search.config.population_size
    assert len(set(expressions)) == len(expressions)
    for node in population:
        validate(node)
        assert EvolutionSearch._depth(node) <= 5
        assert simplify(node) == node


def test_elitism_never_loses_the_best_formula():

    config = EvolutionConfig(
        population_size=8, generations=4, elite_count=2, tournament_size=2,
        max_depth=5, seed=41, validation_top_k=0,
    )
    result = EvolutionSearch(make_research_data(), config).run()

    best = [generation["best_train_score"] for generation in result.generations]
    assert best == sorted(best)


def test_elites_are_carried_into_the_next_generation():

    search = _search()
    scored = search._score_population(search._initial_population())

    population = {str(node) for node in search._next_generation(scored)}

    for elite in scored[: search.config.elite_count]:
        assert elite.expression in population


def test_equivalent_spellings_share_one_evaluation():

    search = _search()
    first = search._score(simplify(parse("ADD(VOLUME,CLOSE)")))
    second = search._score(simplify(parse("ADD(ABS(CLOSE),VOLUME)")))

    assert first is second
    assert len(search._fitness_cache) == 1
    assert search._cache_hits == 1


def test_train_score_rejects_low_coverage_and_undefined_icir():

    assert train_score(_fitness(-0.3, 0.9), min_coverage=0.5) == pytest.approx(0.3)
    assert train_score(_fitness(-0.3, 0.4), min_coverage=0.5) == 0.0
    assert train_score(_fitness(float("nan"), 0.9), min_coverage=0.5) == 0.0
    assert train_score(_fitness(0.3, 0.4), min_coverage=0.0) == pytest.approx(0.3)


def test_low_coverage_formula_scores_zero_in_both_searches():

    # LOG(SUB(OPEN,CLOSE)) is negative everywhere in this data, so it has no values.
    # LOG(SUB(CLOSE,LOW)) is defined everywhere.
    data = make_research_data()
    evolution = EvolutionSearch(data, EvolutionConfig(population_size=8, elite_count=2, min_coverage=0.5))
    random_search = RandomSearch(data, RandomSearchConfig(evaluation_budget=1, min_coverage=0.5))
    sparse = parse("LOG(SUB(OPEN,CLOSE))")

    assert evolution._score(sparse).score == 0.0
    assert evolution._score(sparse).fitness.coverage < 0.5
    candidates = []
    random_search._evaluate(sparse, set(), candidates)
    assert candidates[0].score == 0.0


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_min_coverage_must_be_a_fraction(value):

    with pytest.raises(ValueError, match="min_coverage"):
        EvolutionConfig(min_coverage=value)
    with pytest.raises(ValueError, match="min_coverage"):
        RandomSearchConfig(min_coverage=value)


def test_random_search_deduplicates_equivalent_formulas():

    config = RandomSearchConfig(evaluation_budget=30, seed=7, max_depth=4, validation_top_k=0)
    result = RandomSearch(make_research_data(), config).run()

    expressions = [candidate.expression for candidate in result.best]
    assert all(str(simplify(parse(expression))) == expression for expression in expressions)


def test_seeded_random_draws_do_not_depend_on_simplification():

    # simplify() uses no randomness, so the RNG stream, and hence a seed's
    # sequence of generated trees, is unchanged by canonicalisation.
    a, b = _search(seed=9), _search(seed=9)
    raw = [a._random_tree(5) for _ in range(50)]
    canonical = [simplify(b._random_tree(5)) for _ in range(50)]
    assert [simplify(tree) for tree in raw] == canonical
