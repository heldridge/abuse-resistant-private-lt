import json
import random
from collections import Counter
import argparse

from sage.all import GF, Integer, PolynomialRing, shuffle
from tqdm import tqdm


def serialize_poly(poly, degree):
    return [int(poly[i]) for i in range(degree + 1)]


def serialize_instance(field, pR, n, c, ell, agreement, valid_polys, codeword):
    data = {
        "parameters": {
            "field_size": len(field),
            "variable": str(pR.gens()[0]),
            "n": int(n),
            "c": int(c),
            "ell": int(ell),
            "agreement": int(agreement),
        },
        "present_ranks": {
            int(rank): list(map(lambda poly: serialize_poly(poly, ell), poly_list))
            for rank, poly_list in valid_polys.items()
        },
        "codeword": list(
            map(
                lambda symbol: [
                    int(symbol[0]),
                    list(map(int, symbol[1])),
                    int(symbol[2]),
                ],
                codeword,
            )
        ),
    }
    return data


def load_poly(pR, poly):
    return pR(list(reversed(poly)))


def load_instance(filename):
    with open(filename) as infile:
        data = json.load(infile)

    field = GF(data["parameters"]["field_size"])
    pR = PolynomialRing(field, data["parameters"]["variable"])

    present_ranks = {
        rank: list(map(lambda poly: load_poly(pR, poly), poly_list))
        for rank, poly_list in data["present_ranks"].items()
    }

    codeword = [
        [field(symbol[0]), [field(yc) for yc in symbol[1]], Integer(symbol[2])]
        for symbol in data["codeword"]
    ]

    return field, pR, present_ranks, codeword


def random_non_zero_element(field):
    elt = field.random_element()
    while elt == 0:
        elt = field.random_element()
    return elt


def gen_unique_elements(field, n):
    evaluation_points = []
    for i in range(n):
        elt = field.random_element()
        while elt in evaluation_points:
            elt = field.random_element()
        evaluation_points.append(elt)
    return evaluation_points


def random_polynomial(pR, degree):
    z = pR.gens()[0]
    return pR.random_element(degree=degree - 1) + z ^ degree


def gen_adversarial_instance(field, pR, ell=6, c=3, evals=[12, 12, 1]):
    m = len(evals) - 1  # number of polynomial sets
    n = sum(evals)
    f_list = [[random_polynomial(pR, ell) for i in range(c)] for j in range(m)]
    eval_points = gen_unique_elements(field, n)
    codeword = []
    i = 0
    for k in range(m):
        for j in range(evals[k]):
            codeword.append(
                [eval_points[i], list(fi(eval_points[i]) for fi in f_list[k]), k]
            )
            i += 1
    for j in range(evals[-1]):
        codeword.append(
            [eval_points[i], list(field.random_element() for fi in range(c)), -1]
        )
        i += 1
    return f_list, codeword


def gen_noise_points(field, c, n):
    return [
        [field.random_element(), [field.random_element() for _ in range(c)], -1]
        for _ in range(n)
    ]


def gen_simulated_instance(
    field,
    pR,
    ell,
    c=1,
    num_stalkers=3,
    anonymity_epoch=4,
    delta=4,
    detection_time=3600,
    deletion_percent=5,
    num_noise_points=0,
    previous_windows=23,
):
    if anonymity_epoch % delta != 0:
        raise ValueError(
            f"delta ({delta}) must divide the anonymity epoch ({anonymity_epoch})."
        )
    if detection_time % anonymity_epoch != 0:
        raise ValueError(
            f"anonymity_epoch ({anonymity_epoch}) must divide detection time "
            f"({detection_time})."
        )
    broadcasts_per_anonymity_epoch = anonymity_epoch // delta
    num_anonymity_epochs = detection_time // anonymity_epoch

    used_eval_points = set()
    for _ in range(previous_windows * num_anonymity_epochs):
        used_eval_points.add(random_non_zero_element(field))

    stalker_broadcasts = []
    for stalker in range(num_stalkers):
        # Create a polynomial for the stalker
        stalker_polys = [random_polynomial(pR, ell) for _ in range(c)]
        for _ in range(num_anonymity_epochs):
            # Generate the evaluation point for this anonymity epoch
            eval_point = random_non_zero_element(field)

            if eval_point not in used_eval_points:
                broadcast = [
                    eval_point,
                    [poly(eval_point) for poly in stalker_polys],
                    stalker,
                ]
            else:
                broadcast = [eval_point, [field.random_element() for _ in range(c)], -1]
            used_eval_points.add(eval_point)

            for _ in range(broadcasts_per_anonymity_epoch):
                # Simulate the channel dropping a percentage of broadcasts by only
                # appending the broadcast with a 1 - deletion_percent chance
                if random.randint(1, 100) > deletion_percent:
                    stalker_broadcasts.append(broadcast)

    noise_points = gen_noise_points(field, c, num_noise_points)

    all_broadcasts = stalker_broadcasts + noise_points
    shuffle(all_broadcasts)

    return all_broadcasts


def sample_from_vec(freqs):
    """
    Given a discrete probability distribution `freqs`, sample an item accordingly. 1-indexed to
    match use in Zipfian distributions.

    Args:
        freqs (list[float]): The probability of sampling each value

    Returns:
        int: A value sampled according to the passed frequencies
    """

    rand_num = random.random()
    cumulative_sum = 0.0
    for i, freq in enumerate(freqs):
        cumulative_sum += freq
        if rand_num < cumulative_sum:
            return i + 1


def get_shuffled_sequential_elements(field, start, end):
    elements = [field(e) for e in range(start, end)]
    shuffle(elements)
    return elements


def gen_zipfian_instance(field, pR, ell, c, agreement, support, s, n, tqdm_position=2):
    """
    Generates an instance based on a Zipfian distribution of client inputs

    Args:
        field (FiniteField): The field to draw values from
        pR (PolynomialRing): The polynomial ring clients sample polynomials from
        ell (int): The degree of the polynomials
        c (int): The number of polynomials a client submits points on in each report
        agreement (int): The minimum number of points submitted for the same value at which that
            value should be considered "present" in the input (and therefore need to be
            reconstructed).
        support (int): The support of the Zipfian distribution. The number of possible client
            values.
        s (float): The s parameter for the Zipfian distribution
        n (int): The total number of points in the instance

    Returns:
        tuple[
            dict[int, list[Polynomial]],
            list[
                tuple[Integer, list[Integer], Integer]
            ]
        ]: A generated instance. First tuple element is the present Zipf ranks and their associated
        polynomials. Second is the instance itself. A list of points, each of which also has the
        value associated with it as the final entry.
    """
    normalization_const = sum(1 / (k**s) for k in range(1, support + 1))

    freqs = [(1 / (k**s)) / normalization_const for k in range(1, support + 1)]
    eval_points = get_shuffled_sequential_elements(field, 1, n + 1)

    rank_to_polys = {}
    codeword = []
    rank_counts = Counter()
    for eval_point in tqdm(
        eval_points, desc="Generating instance", leave=False, position=tqdm_position
    ):
        rank = sample_from_vec(freqs)
        rank_counts.update([rank])

        if rank not in rank_to_polys:
            polys = [random_polynomial(pR, ell) for _ in range(c)]
            rank_to_polys[rank] = polys

        codeword.append(
            [eval_point, list(poly(eval_point) for poly in rank_to_polys[rank]), rank]
        )

    present_ranks = {}
    for rank, polys in rank_to_polys.items():
        if rank_counts[rank] >= agreement:
            present_ranks[rank] = polys

    return present_ranks, codeword


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate a Zipfian MDSS instance")
    parser.add_argument("filename", help="The file to write the instance to")
    parser.add_argument(
        "-N",
        help="The number of points in the instance. Default 10,000.",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "-c", help="The MDSS c parameter. Default 20.", type=int, default=20
    )
    parser.add_argument(
        "-t",
        "--agreement",
        help="The number of matching points required for a value to be considered present. Default 143.",
        type=int,
        default=143,
    )
    parser.add_argument(
        "-l",
        "-ell",
        "--ell",
        help="The MDSS ell parameter. Default 20.",
        type=int,
        default=20,
    )
    parser.add_argument(
        "-u",
        "--support",
        help="The support of the Zipfian instance. Default 10,000.",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "-s", help="The Zipf s parameter. Default 1.03.", type=float, default=1.03
    )
    parser.add_argument(
        "--min-field",
        "-mf",
        help="The minimum field size. Default 100,000",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "-v",
        "--variable",
        help="The polynomial variable. Default x.",
        type=str,
        default="x",
    )
    args = parser.parse_args()

    field = GF(Integer(args.min_field).next_prime())
    pR = PolynomialRing(field, args.variable)

    present_ranks, codeword = gen_zipfian_instance(
        field, pR, args.ell, args.c, args.agreement, args.support, args.s, args.N, 0
    )

    serialized = serialize_instance(
        field, pR, args.N, args.c, args.ell, args.agreement, present_ranks, codeword
    )

    with open(args.filename, "w+") as outfile:
        json.dump(serialized, outfile)
