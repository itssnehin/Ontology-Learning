from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal
import re
import difflib

def code_parts(code):
    m = re.match(r"([A-Z]+)(\d+)", code)
    return (m.group(1), int(m.group(2))) if m else (code, None)

def generate_iec_ontology(input_txt, owl_out, ttl_out, base_iri="http://example.org/iec61360#"):
    g = Graph()
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    IEC = Namespace(base_iri)
    g.bind("iec", IEC)

    parent_stack = []

    # Read taxonomy file
    with open(input_txt, "r", encoding="utf-8") as f:
        entries = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"([A-Z0-9/]+)\s*-\s*(.+)", line)
            if match:
                entries.append((match.group(1), match.group(2)))

    # Build ontology with semantic + numeric rules
    for code, label in entries:
        alpha, num = code_parts(code)
        class_uri = IEC[code]

        # Add class and label
        g.add((class_uri, RDF.type, OWL.Class))
        g.add((class_uri, RDFS.label, Literal(label)))

        # Try to find a valid parent
        for parent_code in reversed(parent_stack):
            pa, pn = code_parts(parent_code)
            parent_label = next((lab for c, lab in entries if c == parent_code), "").lower()

            if pa != alpha:
                break  # Different alpha prefix → stop upward search

            sim = difflib.SequenceMatcher(None, label.lower(), parent_label).ratio()

            # Numeric closeness and semantic relevance
            if pn is not None and num is not None and num > pn and sim >= 0.2:
                g.add((class_uri, RDFS.subClassOf, IEC[parent_code]))
                break

        parent_stack.append(code)

    # Save OWL & TTL
    g.serialize(destination=owl_out, format="xml")
    g.serialize(destination=ttl_out, format="turtle")
    print(f"Ontology saved: {owl_out}, {ttl_out}")

# Example usage:
# generate_iec_ontology("./data/iec_61360_4.txt", "./data/iec_61360_clean.owl", "./data/iec_61360_clean.ttl")
