"""
London Danger Database
======================
Edge danger scores ∈ [0, 1] for pedestrian risk on the London map.

METHODOLOGY:
  Scores are weighted toward violent crime and robbery rather than total
  crime count. Tourist areas like Westminster get LOW scores despite high
  total crime, because most of that crime is pickpocketing under heavy
  CCTV — the per-pedestrian violent risk is low. Outer boroughs in
  Haringey, Lambeth, and Croydon get HIGH scores because their violent
  crime rates per resident are elevated.

DATA SOURCES:
  - data.police.uk — UK Police open data portal (Open Government Licence v3.0)
  - safeareaslondon.com — 500×500m analytical grid, street-level crime density
  - crimerate.co.uk — severity-weighted Crime Risk Scores by borough
  - Metropolitan Police Crime Dashboard (London City Hall)
  Data period: January 2024 – November 2025.

This file is consumed by safety_aware_a_star.load_map("london").
"""

# Each tuple: (node_u, node_v, danger_score)
# Edges are undirected — the algorithm builds both (u,v) and (v,u) entries.
EDGE_DANGERS = [
    # =====================================================================
    # CENTRAL ZONE — well-policed, well-lit, low pedestrian risk
    # Heavy CCTV and police presence suppress violent crime despite high
    # tourist footfall. Most edges in this zone score 0.05–0.20.
    # =====================================================================
    (0,  1,  0.10),   # Westminster → Covent Garden: tourist corridor
    (0,  3,  0.08),   # Westminster → South Bank: well-lit riverside
    (0,  4,  0.12),   # Westminster → Soho: nightlife, slightly elevated
    (0,  7,  0.08),   # Westminster → Waterloo: well-policed station
    (0,  8,  0.06),   # Westminster → Marylebone: affluent residential
    (0,  13, 0.08),   # Westminster → Notting Hill: affluent west corridor
    (0,  16, 0.15),   # Westminster → Clapham: crosses Vauxhall
    (0,  24, 0.35),   # Westminster → Brixton: crosses Lambeth
    (1,  2,  0.08),   # Covent Garden → City of London
    (1,  3,  0.07),   # Covent Garden → South Bank: Waterloo Bridge
    (1,  4,  0.10),   # Covent Garden → Soho: adjacent, busy
    (1,  9,  0.06),   # Covent Garden → Bloomsbury: quiet academic area
    (2,  3,  0.08),   # City of London → South Bank: London Bridge
    (2,  6,  0.07),   # City of London → Tower Bridge
    (2,  11, 0.18),   # City of London → Shoreditch: transitions to higher
    (2,  14, 0.12),   # City of London → Islington: gentrified
    (3,  6,  0.10),   # South Bank → Tower Bridge: riverside path
    (3,  7,  0.06),   # South Bank → Waterloo: adjacent, well-lit
    (3,  12, 0.25),   # South Bank → Elephant & Castle: crime increases
    (4,  8,  0.08),   # Soho → Marylebone: quiet transition
    (4,  9,  0.07),   # Soho → Bloomsbury: safe academic streets
    (4,  10, 0.20),   # Soho → Camden: nightlife corridor, theft rises
    (4,  13, 0.10),   # Soho → Notting Hill: through affluent areas
    (5,  9,  0.10),   # Kings Cross → Bloomsbury: station area
    (5,  10, 0.22),   # Kings Cross → Camden: known theft/drug hotspot
    (5,  14, 0.15),   # Kings Cross → Islington: mixed
    (5,  20, 0.55),   # Kings Cross → Tottenham: direct north, high crime
    (5,  26, 0.50),   # Kings Cross → Wood Green: through Haringey
    (6,  11, 0.20),   # Tower Bridge → Shoreditch: crime increases east
    (6,  12, 0.22),   # Tower Bridge → Elephant & Castle: south, mixed
    (6,  15, 0.12),   # Tower Bridge → Greenwich: safe-ish riverside
    (6,  25, 0.35),   # Tower Bridge → Peckham: through Southwark
    (7,  12, 0.25),   # Waterloo → Elephant & Castle: rougher
    (7,  16, 0.15),   # Waterloo → Clapham: south through residential
    (7,  24, 0.38),   # Waterloo → Brixton: Lambeth corridor
    (8,  9,  0.06),   # Marylebone → Bloomsbury: safe central
    (8,  10, 0.18),   # Marylebone → Camden: transition
    (8,  13, 0.05),   # Marylebone → Notting Hill: affluent west
    (8,  17, 0.08),   # Marylebone → Hammersmith: safe west corridor
    (9,  10, 0.18),   # Bloomsbury → Camden: near Euston
    (9,  14, 0.12),   # Bloomsbury → Islington: short, mixed

    # =====================================================================
    # INNER RING — mixed, gentrified next to deprived
    # Adjacent 500m grid blocks can have very different Local Crime Level
    # values according to safeareaslondon.com.
    # =====================================================================
    (10, 14, 0.20),   # Camden → Islington: drug/theft issues
    (10, 20, 0.55),   # Camden → Tottenham: into high crime zone
    (10, 26, 0.50),   # Camden → Wood Green: through Haringey
    (11, 14, 0.18),   # Shoreditch → Islington: gentrified, bike theft
    (11, 18, 0.40),   # Shoreditch → Hackney: elevated violent crime
    (11, 19, 0.30),   # Shoreditch → Stratford: east, mixed
    (12, 15, 0.25),   # Elephant → Greenwich: south-east
    (12, 16, 0.22),   # Elephant → Clapham: south residential
    (12, 22, 0.40),   # Elephant → Lewisham: elevated knife crime
    (12, 24, 0.45),   # Elephant → Brixton: Lambeth high-crime corridor
    (12, 25, 0.40),   # Elephant → Peckham: Southwark, elevated
    (13, 17, 0.08),   # Notting Hill → Hammersmith: safe west London
    (14, 18, 0.35),   # Islington → Hackney: increases sharply eastward
    (15, 22, 0.25),   # Greenwich → Lewisham: south-east, moderate
    (15, 25, 0.30),   # Greenwich → Peckham: through Southwark
    (15, 27, 0.28),   # Greenwich → Woolwich: further east, mixed
    (16, 21, 0.50),   # Clapham → Croydon: into high-crime borough
    (16, 24, 0.35),   # Clapham → Brixton: Lambeth corridor
    (18, 19, 0.35),   # Hackney → Stratford: east London, mixed
    (18, 20, 0.60),   # Hackney → Tottenham: north-east, both high
    (18, 23, 0.55),   # Hackney → Barking: deprived east corridor
    (18, 29, 0.55),   # Hackney → Seven Sisters: Haringey
    (19, 23, 0.45),   # Stratford → Barking: deprived
    (19, 27, 0.35),   # Stratford → Woolwich: Crossrail corridor

    # =====================================================================
    # OUTER ZONE — high crime corridors
    # Haringey (Tottenham, Wood Green, Seven Sisters), Lambeth, and Croydon
    # consistently appear in the top brackets for violent crime per resident.
    # =====================================================================
    (20, 26, 0.70),   # Tottenham → Wood Green: both Haringey
    (20, 28, 0.75),   # Tottenham → Edmonton: north Haringey/Enfield
    (20, 29, 0.65),   # Tottenham → Seven Sisters: Haringey corridor
    (21, 22, 0.55),   # Croydon → Lewisham: south-east link
    (21, 25, 0.50),   # Croydon → Peckham: south London corridor
    (22, 25, 0.45),   # Lewisham → Peckham: borough boundary
    (22, 27, 0.40),   # Lewisham → Woolwich: south-east
    (23, 27, 0.45),   # Barking → Woolwich: deprived east
    (24, 25, 0.42),   # Brixton → Peckham: Lambeth/Southwark
    (26, 28, 0.70),   # Wood Green → Edmonton: Haringey/Enfield
    (26, 29, 0.65),   # Wood Green → Seven Sisters: Haringey corridor
    (28, 29, 0.60),   # Edmonton → Seven Sisters: outer north
]