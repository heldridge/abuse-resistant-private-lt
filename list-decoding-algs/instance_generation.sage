import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sage.all import GF, Integer, PolynomialRing, ceil, shuffle
from tqdm import tqdm


def mdss_threshold(n, c, ell):
    return ceil((n + (c * ell)) / (c + 1))


def serialize_poly(poly, degree):
    return [int(poly[i]) for i in range(degree + 1)]


def load_poly(pR, poly):
    return pR(list(reversed(poly)))


@dataclass
class Instance:
    field: Any
    pR: Any
    n: int
    c: int
    ell: int
    agreement: int
    present_dealers: dict[int, list[list[Any]]]
    codeword: list[tuple[Integer, list[Integer]]]
    symbol_to_dealer: list[int]
    is_nice: bool = field(init=False)

    @staticmethod
    def calc_if_nice(threshold, symbol_to_dealer):

        if len(symbol_to_dealer) == 0:
            return True

        c = Counter(symbol_to_dealer)

        prev_count = c.most_common(1)[0][1] + 2

        for _, count in c.most_common():
            if count < threshold:
                break
            if prev_count - count <= 1 or count == threshold:
                return False

            prev_count = count

        return True

    def __post_init__(self):
        threshold = mdss_threshold(self.n, self.c, self.ell)
        self.is_nice = Instance.calc_if_nice(threshold, self.symbol_to_dealer)

    def to_serializable(self):
        return {
            "parameters": {
                "field_size": len(self.field),
                "variable": str(self.pR.gens()[0]),
                "n": int(self.n),
                "c": int(self.c),
                "ell": int(self.ell),
                "agreement": int(self.agreement),
                "is_nice": self.is_nice,
            },
            "present_dealers": {
                int(rank): list(
                    map(lambda poly: serialize_poly(poly, self.ell), poly_list)
                )
                for rank, poly_list in self.present_dealers.items()
            },
            "codeword": list(
                map(
                    lambda symbol: [
                        int(symbol[0]),
                        list(map(int, symbol[1])),
                    ],
                    self.codeword,
                )
            ),
            "symbol_to_dealer": list(map(int, self.symbol_to_dealer)),
        }

    @staticmethod
    def from_json(data):
        params = data["parameters"]
        f = GF(params["field_size"])
        pR = PolynomialRing(f, params["variable"])

        return Instance(
            f,
            pR,
            params["n"],
            params["c"],
            params["ell"],
            params["agreement"],
            {
                dealer_id: list(map(lambda poly: load_poly(pR, poly), poly_list))
                for dealer_id, poly_list in data["present_dealers"].items()
            },
            [
                [f(symbol[0]), [f(yc) for yc in symbol[1]]]
                for symbol in data["codeword"]
            ],
            data["symbol_to_dealer"],
        )


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


def gen_zipfian_instance(field, pR, ell, c, agreement, support, s, n, **tqdm_args):
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
    symbol_to_dealer = []
    for eval_point in tqdm(eval_points, **tqdm_args):
        rank = sample_from_vec(freqs)

        if rank not in rank_to_polys:
            polys = [random_polynomial(pR, ell) for _ in range(c)]
            rank_to_polys[rank] = polys

        codeword.append(
            [eval_point, list(poly(eval_point) for poly in rank_to_polys[rank])]
        )
        symbol_to_dealer.append(rank)

    present_ranks = {}
    rank_counts = Counter(symbol_to_dealer)

    for rank, polys in rank_to_polys.items():
        if rank_counts[rank] >= agreement:
            present_ranks[rank] = polys

    return Instance(
        field, pR, n, c, ell, agreement, present_ranks, codeword, symbol_to_dealer
    )


def gen_malicious_client_instance(field, pR, ell, c, agreement, evals, **tqdm_args):
    """
    Generates an instance where there is a maliciously crafted input point.
    Malicious input is of the form (x, y_1, y_2,..., r,..., y_c), where y_i is an honestly
    generated point, and r is a randomly sampled value.
    """
    n = sum(evals)
    rank_to_polys = [
        [random_polynomial(pR, ell) for _ in range(c)] for _ in range(len(evals))
    ]

    eval_points = gen_unique_elements(field, n)

    i = 0
    codeword = []
    symbol_to_dealer = []
    done_bad_point = False
    for dealer, count in enumerate(evals):
        eval_point = eval_points[i]

        for _ in range(count):
            point = [
                eval_point,
                list(poly(eval_point) for poly in rank_to_polys[dealer]),
            ]
            if not done_bad_point:
                point[1][0] = field.random_element()

            codeword.append(point)
            symbol_to_dealer.append(dealer)
        i += 1

    present_ranks = {}
    rank_counts = Counter(symbol_to_dealer)
    for dealer, polys in enumerate(rank_to_polys):
        if rank_counts[dealer] >= agreement:
            present_ranks[dealer] = polys

    return Instance(
        field, pR, n, c, ell, agreement, present_ranks, codeword, symbol_to_dealer
    )


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
    parser.add_argument(
        "-m",
        "--malicious",
        help="Generate a malicious rather than Zipfian instance",
        action="store_true",
    )
    parser.add_argument(
        "-e",
        "--evals",
        nargs="*",
        help="The number of points each malicious client should contribute",
        type=int,
    )  # TODO: Replace with a subcommand rather than two flags
    args = parser.parse_args()

    field = GF(Integer(args.min_field).next_prime())
    pR = PolynomialRing(field, args.variable)

    if args.malicious:
        inst = gen_malicious_client_instance(
            field, pR, args.ell, args.c, args.agreement, args.evals
        )
    else:
        inst = gen_zipfian_instance(
            field, pR, args.ell, args.c, args.agreement, args.support, args.s, args.N
        )

    with open(args.filename, "w+") as outfile:
        json.dump(inst.to_serializable(), outfile)
