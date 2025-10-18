### Category 1: Find Concepts Based on Attributes (Filtering & Discovery)

These queries are used to find specific components or classes based on their known characteristics and relationships.

**1. Find a Concept by Exact Name**
*   **Use Case:** Verify if a specific, known component class exists in the ontology.
*   **Query:**
    ```cypher
    MATCH (c:OntologyClass {name: 'FPCAntenna'})
    RETURN c.name, c.source, c.uri;
    ```
*   **Explanation:** This is the most basic lookup to retrieve the core details of a single concept.

**2. Find Concepts Using a Partial or Fuzzy Name Search**
*   **Use Case:** The engineer knows part of a component's name but not the exact term.
*   **Query:**
    ```cypher
    MATCH (c:OntologyClass)
    WHERE c.name CONTAINS 'Antenna'
    RETURN c.name;
    ```
*   **Explanation:** Uses `CONTAINS` for substring matching, useful for discovery when the exact name isn't known.

**3. Find All Direct Subclasses of a Parent Concept**
*   **Use Case:** "I need a `PassiveComponent`. What are my options?"
*   **Query:**
    ```cypher
    MATCH (child)-[:SUBCLASS_OF]->(parent:OntologyClass {name: 'PassiveComponent'})
    RETURN child.name;
    ```
*   **Explanation:** This retrieves the direct children in the hierarchy, helping the engineer navigate the taxonomy.

**4. Find All Descendants of a Parent Concept (Transitive Query)**
*   **Use Case:** "Show me *everything* that falls under the category of `Antenna`, no matter how deeply nested."
*   **Query:**
    ```cypher
    MATCH (descendant)-[:SUBCLASS_OF*1..]->(parent:OntologyClass {name: 'Antenna'})
    RETURN descendant.name;
    ```
*   **Explanation:** The `*1..` syntax finds all nodes connected by one or more `:SUBCLASS_OF` relationships, providing a complete view of a branch.

**5. Find Concepts by Non-Taxonomic Relationship**
*   **Use Case:** "What components are known to be `PARTOF` an `IntegratedCircuit`?"
*   **Query:**
    ```cypher
    MATCH (part)-[:PARTOF]->(whole:OntologyClass {name: 'IntegratedCircuit'})
    RETURN part.name;
    ```
*   **Explanation:** This leverages the learned non-taxonomic relations to discover functional or compositional connections.

**6. Find Concepts with Multiple Attributes (Combined Filtering)**
*   **Use Case:** "I need a component that is a `SUBCLASS_OF` `RFComponent` AND is also `RELATEDTO` `Electromechanical` components."
*   **Query:**
    ```cypher
    MATCH (c:OntologyClass)-[:SUBCLASS_OF]->(:OntologyClass {name: 'RFComponent'})
    MATCH (c)-[:RELATEDTO]->(:OntologyClass {name: 'Electromechanical'})
    RETURN c.name;
    ```
*   **Explanation:** Demonstrates complex filtering by combining multiple relationship-based criteria.

**7. Find Concepts Learned from the Dataset (Exclude Baseline)**
*   **Use Case:** "Show me only the new component classes that the pipeline discovered, not the baseline schema."
*   **Query:**
    ```cypher
    MATCH (c:OntologyClass)
    WHERE c.source = 'learned_from_dataset'
    RETURN c.name;
    ```
*   **Explanation:** Filters concepts based on their origin, which is crucial for evaluating the learning process itself.

---

### Category 2: Enumerate Values (Listing & Aggregation)

These queries are used to get lists of available concepts, types, and other aggregate information.

**8. List All Top-Level Component Families**
*   **Use Case:** "What are the main categories of components defined in this ontology?"
*   **Query:**
    ```cypher
    MATCH (family)-[:SUBCLASS_OF]->(root:OntologyClass {name: 'ElectronicComponent'})
    RETURN family.name ORDER BY family.name;
    ```
*   **Explanation:** Provides a high-level overview of the domain's primary branches.

**9. List All Unique (Non-Taxonomic) Relationship Types Learned**
*   **Use Case:** "What kinds of relationships does our system know about, besides 'is a'?"
*   **Query:**
    ```cypher
    MATCH ()-[r]->()
    WHERE NOT type(r) = 'SUBCLASS_OF'
    RETURN DISTINCT type(r) AS relationship_type;
    ```
*   **Explanation:** This meta-query inspects the schema of the graph itself to show the richness of the learned relations.

**10. List All Concepts Marked for Manual Review**
*   **Use Case:** As a knowledge engineer, "What are the new concepts that the pipeline was uncertain about?"
*   **Query:**
    ```cypher
    MATCH (c:NeedsReview)
    RETURN c.name;
    ```
*   **Explanation:** Directly supports the human-in-the-loop validation workflow.

**11. List All "Leaf" Concepts in a Category**
*   **Use Case:** "Show me the most specific types of `Capacitor` we have defined (those with no subclasses)."
*   **Query:**
    ```cypher
    MATCH (leaf)-[:SUBCLASS_OF*1..]->(:OntologyClass {name: 'Capacitor'})
    WHERE NOT (()-[:SUBCLASS_OF]->(leaf))
    RETURN leaf.name;
    ```
*   **Explanation:** Identifies the terminal nodes in a branch of the hierarchy, representing the most granular concepts.

**12. List the Top 10 Most Connected "Hub" Concepts**
*   **Use Case:** "What are the most central or important concepts in our ontology?"
*   **Query:**
    ```cypher
    MATCH (c:OntologyClass)
    WITH c, size((c)--()) AS degree
    RETURN c.name, degree
    ORDER BY degree DESC
    LIMIT 10;
    ```
*   **Explanation:** Helps identify foundational concepts that are highly interconnected, like `ElectronicComponent` or `IntegratedCircuit`.

---

### Category 3: Focus on Concepts (Hierarchy & Structure Exploration)

These queries explore the structural properties of the ontology itself.

**13. Show the Full Taxonomic Path for a Specific Concept**
*   **Use Case:** "For `FPCAntenna`, show me its entire classification hierarchy up to the root."
*   **Query:**
    ```cypher
    MATCH path = (leaf:OntologyClass {name: 'FPCAntenna'})-[:SUBCLASS_OF*]->(root:OntologyClass {name: 'Thing'})
    RETURN [node in nodes(path) | node.name] AS hierarchy;
    ```
*   **Explanation:** Traces a concept's lineage up the `SUBCLASS_OF` tree, which is fundamental for understanding its classification.

**14. Find "Sibling" Concepts**
*   **Use Case:** "What are other types of `Antenna` that are at the same level as `ChipAntenna`?"
*   **Query:**
    ```cypher
    MATCH (sibling)-[:SUBCLASS_OF]->(parent)<-[:SUBCLASS_OF]-(:OntologyClass {name: 'ChipAntenna'})
    WHERE sibling.name <> 'ChipAntenna'
    RETURN sibling.name;
    ```
*   **Explanation:** Finds concepts that share the same direct parent, useful for finding alternatives or related components.

**15. Show All Outgoing Non-Taxonomic Relationships for a Concept**
*   **Use Case:** "Tell me everything you know about a `cable assembly` besides what it is a subclass of."
*   **Query:**
    ```cypher
    MATCH (c:OntologyClass {name: 'cable assembly'})-[r]->(t:OntologyClass)
    WHERE NOT type(r) = 'SUBCLASS_OF'
    RETURN type(r) AS relation, t.name AS target;
    ```
*   **Explanation:** Provides a complete profile of a concept's functional, compositional, and other learned relationships.

**16. Find the Shortest Path Between Two Concepts**
*   **Use Case:** "How are `Resistor` and `Antenna` conceptually related in this ontology?"
*   **Query:**
    ```cypher
    MATCH a_node = (a:OntologyClass {name: 'Resistor'}), b_node = (b:OntologyClass {name: 'Antenna'}),
          p = shortestPath((a)-[*]-(b))
    RETURN p;
    ```
*   **Explanation:** An advanced query that reveals the chain of relationships connecting two seemingly disparate concepts, highlighting implicit knowledge.

**17. Find Concepts that Bridge Two Major Categories**
*   **Use Case:** "Are there any components that are related to both `ActiveComponent` and `RFComponent`?"
*   **Query:**
    ```cypher
    MATCH (a:OntologyClass)-[:SUBCLASS_OF*]->(:OntologyClass {name: 'ActiveComponent'})
    MATCH (a)-[*]-(b)
    MATCH (b)-[:SUBCLASS_OF*]->(:OntologyClass {name: 'RFComponent'})
    RETURN DISTINCT a.name AS active_related, b.name AS rf_related;
    ```
*   **Explanation:** Helps discover cross-domain concepts or components that serve multiple functions.

**18. Find the Direct Parent of a Concept**
*   **Use Case:** "What is the immediate parent class for `ChipAntenna`?"
*   **Query:**
    ```cypher
    MATCH (child {name: 'ChipAntenna'})-[:SUBCLASS_OF]->(parent)
    RETURN parent.name;
    ```
*   **Explanation:** A simple but essential query for navigating the hierarchy step-by-step.

**19. List All Relationships (Both Ways) for a Concept**
*   **Use Case:** "Give me a complete 360-degree view of all connections to and from the `Capacitor` concept."
*   **Query:**
    ```cypher
    MATCH (c:OntologyClass {name: 'Capacitor'})-[r]-(t:OntologyClass)
    RETURN c.name, type(r), t.name;
    ```
*   **Explanation:** Uses an undirected relationship match `()`-`[]`-`()` to find both incoming and outgoing relationships.

---

### Category 4: Look at the Coverage (Meta-Analysis & Auditing)

These queries assess the state, quality, and completeness of the learned ontology.

**20. Count Concepts per Top-Level Family**
*   **Use Case:** "How balanced is our ontology? Give me a count of specific concepts under each main family."
*   **Query:**
    ```cypher
    MATCH (family)-[:SUBCLASS_OF]->(:OntologyClass {name: 'ElectronicComponent'})
    MATCH (leaf)-[:SUBCLASS_OF*1..]->(family)
    WHERE NOT (()-[:SUBCLASS_OF]->(leaf))
    RETURN family.name, count(DISTINCT leaf) AS specific_concept_count
    ORDER BY specific_concept_count DESC;
    ```
*   **Explanation:** Provides a summary of which areas of the ontology are most detailed.

**21. Count Relationship Types (Taxonomic vs. Non-Taxonomic)**
*   **Use Case:** "What is the ratio of hierarchical knowledge vs. functional knowledge in our graph?"
*   **Query:**
    ```cypher
    MATCH ()-[r]->()
    RETURN CASE WHEN type(r) = 'SUBCLASS_OF' THEN 'Taxonomic' ELSE 'Non-Taxonomic' END AS relation_category, count(r) AS count;
    ```
*   **Explanation:** Gives a high-level statistical overview of the ontology's structure.

**22. List Concepts with No Non-Taxonomic Relationships**
*   **Use Case:** "Which concepts have been classified but have no other descriptive relationships? These might need more detail."
*   **Query:**
    ```cypher
    MATCH (c:OntologyClass)
    WHERE c.source = 'learned_from_dataset' AND size([(c)-[r]-() WHERE NOT type(r) = 'SUBCLASS_OF' | r]) = 0
    RETURN c.name;
    ```
*   **Explanation:** An excellent auditing query to find "under-described" concepts that may require further enrichment.

**23. Find the "Deepest" Concepts in the Hierarchy**
*   **Use Case:** "Which concepts are the most specialized in our ontology?"
*   **Query:**
    ```cypher
    MATCH path = (leaf)-[:SUBCLASS_OF*]->(:OntologyClass {name: 'Thing'})
    WHERE NOT (()-[:SUBCLASS_OF]->(leaf))
    RETURN leaf.name, length(path) AS depth
    ORDER BY depth DESC
    LIMIT 10;
    ```
*   **Explanation:** Measures specialization by finding the concepts furthest down the `SUBCLASS_OF` chain.

**24. Find "Orphan" Concepts Not Connected to the Main Hierarchy**
*   **Use Case:** "Did the pipeline learn any concepts that it failed to connect to the 'Thing' root? This would indicate a problem."
*   **Query:**
    ```cypher
    MATCH (c:OntologyClass)
    WHERE c.source = 'learned_from_dataset' AND NOT (c)-[:SUBCLASS_OF*]->(:OntologyClass {name: 'Thing'})
    RETURN c.name;
    ```
*   **Explanation:** A critical integrity check to ensure all learned concepts are properly integrated into the main taxonomic tree.

**25. What are the Most Common Non-Taxonomic Relationship Types?**
*   **Use Case:** "What kind of associative knowledge is our pipeline best at finding?"
*   **Query:**
    ```cypher
    MATCH ()-[r]->()
    WHERE NOT type(r) = 'SUBCLASS_OF'
    RETURN type(r) AS relationship, count(r) AS frequency
    ORDER BY frequency DESC
    LIMIT 10;
    ```
*   **Explanation:** Helps understand the patterns and biases in the non-taxonomic relation extraction.

**26. Find Categories with the Most Items Needing Review**
*   **Use Case:** "Which area of the ontology is the pipeline most uncertain about? I should focus my review efforts there."
*   **Query:**
    ```cypher
    MATCH (c:NeedsReview)-[:SUBCLASS_OF*1..]->(family)-[:SUBCLASS_OF]->(:OntologyClass {name: 'ElectronicComponent'})
    RETURN family.name, count(c) AS review_count
    ORDER BY review_count DESC;
    ```
*   **Explanation:** Prioritizes the manual review process by identifying hotspots of uncertainty.

**27. List All Concepts and Their Direct Parent**
*   **Use Case:** "Generate a simple, two-column list of all learned concepts and their parent class for a quick review."
*   **Query:**
    ```cypher
    MATCH (child:OntologyClass)-[:SUBCLASS_OF]->(parent:OntologyClass)
    WHERE child.source = 'learned_from_dataset'
    RETURN child.name, parent.name
    ORDER BY parent.name, child.name;
    ```*   **Explanation:** Provides a flat, readable list of all taxonomic links for easy auditing.

**28. Find Concepts with Potentially Redundant `RELATEDTO` Links**
*   **Use Case:** "Are there any concepts that are linked via `SUBCLASS_OF` *and* `RELATEDTO`? The `RELATEDTO` link is likely redundant."
*   **Query:**
    ```cypher
    MATCH (a:OntologyClass)-[:SUBCLASS_OF]-(b:OntologyClass)
    MATCH (a)-[:RELATEDTO]-(b)
    RETURN a.name, b.name;
    ```
*   **Explanation:** An integrity/cleanup query to find and potentially prune unnecessary relationships.

**29. Count the number of relationships for each type**
*   **Use Case:** To see a complete breakdown of all relationship types and their frequencies.
*   **Query:**
    ```cypher
    MATCH ()-[r]->()
    RETURN type(r) AS relationship_type, count(r) AS count
    ORDER BY count DESC;
    ```
*   **Explanation:** This gives a full statistical profile of the graph