# Prompt Templates

## Prompt template for Task 1: Area Description

```text
You are a highly qualified expert trained to caption remote sensing images. Your task is to produce a single, coherent paragraph of about 50 words describing the chosen element in the image. You will be given relevant OpenStreetMap (OSM) information. Follow the instructions below to complete the task.

# Instruction

1. **Description Focus**:
    - Describe the element naturally as a feature within the scene without referencing it as a “selected area” or using similar phrasing (e.g., avoid “this selected area,” “this region,” or “this element”). Instead, simply describe what is visible: "A rectangular area..." or "A broad farmland..."
    - Emphasize its position (e.g., “at the right-center of the scene”), shape (e.g., “a rectangular area”), approximate size, and key visually significant attributes or land uses.
    - If the element consists of multiple polygons, mention each polygon's location and shape within the scene.
    - Integrate direct interpretations from the provided data and any reasonable inferences.
    - Where applicable, include brief references to surroundings (e.g., adjacent vegetation or structures) if it can be reasonably inferred from the data or context. If the data does not confirm their presence, use cautious language (e.g., “probably,” “possibly,” “likely”).

2. **Use of Provided OSM Data**:
    - Convert coordinates and area fractions into qualitative descriptors (e.g., “covering more than half of the view”).
    - Present factual information from the tags directly and confidently. For example, if the data says it is a residential neighborhood called Inwood Forest, do not say “likely” or “appears to be.” State it as a fact.
    - Only introduce uncertainty or inference if the data does not directly confirm something.
    - Do not quote or explicitly present raw OSM keys, tag names, or tag values in the caption. For instance, if a tag identifies the element as a river, simply describe it as a river. 
    - Refrain from using phrases like “tagged as” or “the tag indicates.”

3. **Inferences and Naming Conventions**:
    - Include uncertain or inferred details in a natural way.if you must speculate about the type of buildings, say “possibly featuring single-family homes.”
    - Use any real-world place names mentioned in the tags as they are (e.g., Sacramento County, Inwood Forest).

4. **Coordinate and Orientation Note**:
    - The coordinate system is normalized with (0, 0) at the bottom-left and (1, 1) at the top-right. In this system, the left corresponds to the east, and the right to the west. The area of the entire Region of Interest (ROI) is normalized to 1.

5. **Fluency**:
    - The paragraph should remain fluent and logically cohesive.
    - Aim for around 50 words.

6. **Output Language Flexibility**:
    - Write exclusively in English, avoiding non-English characters or symbols.
    - Avoid using special characters like underscores in the caption.
    - While examples are provided for guidance, you are encouraged to use varied and natural phrasing that best fits the context, ensuring clarity and simplicity.

# Examples

## Example 1

Raw:
The selected area locates at the center of the image patch. It has a irregular shape. It occupies 0.991 of the image patch. Its smoothed boundary is {{[(0.830, -0.000), (0.000, 0.025), (0.000, 1.000), (1.000, 1.000), (0.830, -0.000)]}}.
Some parts of the geometry extend beyond this ROI.
It has following tags:
Tag 1. Its key is 'acres' and value is '74.3587018241'. The tag can be interpreted by its key and value
Tag 2. Its key is 'addr:county', which means The county indicated in the address. The tag belongs to a tag group 'addresses'. The tag value is Sacramento.
Tag 3. Its key is 'attribution', which means The data source of the information used to tag and/or place this element (often considered deprecated). The tag belongs to a tag group 'annotations'. The tag value is Farmland Mapping and Monitoring Program.
Tag 4. Its key is 'description', which means To provide additional information about the related element to the end map user. The tag belongs to a tag group 'annotations'. The tag value is farmland of local importance.
Tag 5. 'landuse:farmland'. The tag belongs to a tag group ''. This tag means: Area of farmland used mainly for tillage (annual crops).
Tag 6. Its key is 'type', which means Type of a relation. The tag belongs to a tag group 'properties'. The tag value is multipolygon.

Caption:
Stretching across nearly the entire scene, an irregular expanse of farmland anchors the central view, encompassing roughly 74 acres and designated as an area of local importance within Sacramento County, likely devoted to annual crops.

## Example 2

Raw:
The selected area locates at the right-center of the image patch. It has a rectangular shape. It occupies 0.537 of the image patch. Its smoothed boundary is {{[(0.375, 0.871), (1.000, 0.861), (1.000, 0.000), (0.385, 0.000), (0.375, 0.871)]}}.
Some parts of the geometry extend beyond this ROI.
It has following tags:
Tag 1. 'landuse:residential'. The tag belongs to a tag group ''. This tag means: An area with predominantly houses or apartment buildings.
Tag 2. Its key is 'name', which means The primary name: in general, the most prominent signposted name or the most common name in the local language(s). The tag belongs to a tag group 'names'. The tag value is Inwood Forest.
Tag 3. 'place:neighbourhood'. The tag belongs to a tag group ''. This tag means: A neighbourhood is a smaller named, geographically localised place within a suburb of a larger city or within a town or village

Caption:
At the right-center of the scene, a rectangular zone occupies more than half the view, designated as Inwood Forest, a residential neighborhood. Its developed landscape includes homes or apartments, perhaps arranged along tree-lined streets suggested by the name, forming a distinctly urban setting likely within a larger municipality.

## Example 3

Raw:
The selected area locates at the bottom-center of the image patch. It has a irregular shape. It occupies 0.185 of the image patch. Its smoothed boundary is {{[(0.866, 0.130), (0.428, -0.000), (0.330, 0.523), (0.384, 0.552), (0.866, 0.130)]}}.
Some parts of the geometry extend beyond this ROI.
It has following tags:
Tag 1. 'natural:wood'. The tag belongs to a tag group 'natural'. This tag means: Tree-covered area (a 'forest' or 'wood')

Caption:
An irregularly shaped wooded area stretches across the bottom-center of the scene, covering nearly one-fifth of the view. Its sinuous boundary hints at a forest likely dense with mature trees that extends beyond the frame suggesting vast, continuous woodland.

## Example 4

Raw:
The selected area locates at the left-bottom, right-bottom of the image patch. It has a multi-polygonal shape. It occupies 0.206 of the image patch. Its smoothed boundary is{{[(0.228, 0.192), (0.424, 0.317), (0.435, 0.000), (0.000, 0.000), (0.228, 0.192)].[(0.686, 0.339), (1.000, 0.286), (1.000, 0.000),(0.695, 0.000),(0.686, 0.339)]}}.
Some parts of the geometry extend beyond this ROI.
It has following tags:
Tag 1. 'landuse:commercial'. The tag belongs to a tag group ''. This tag means: A commercial zone, predominantly offces or services.

Caption:
At the lower part of the scene, two polygonal blocks define a commercial zone of offices or services, together covering about one-fifth of the view. One polygon stretches across the lower-left side, while another occupies the lower-right corner, both partially extending beyond the frame suggesting a developed urban setting.

## Example 5

Raw:
The selected area locates at the top-center of the image patch. It has a irregular shape. It occupies 0.490 of the image patch. Its smoothed boundary is {{[(0.987, 0.571), (0.000, 0.527), (0.000, 1.000), (1.000, 1.000), (0.987, 0.571)]}}.
Some parts of the geometry extend beyond this ROI.
It has following tags:
Tag 1. Its key is 'name', which means The primary name: in general, the most prominent signposted name or the most common name in the local language(s). The tag belongs to a tag group 'names'. The tag value is West Lake.
Tag 2. 'natural:water'. The tag belongs to a tag group 'natural'. This tag means: Any inland body of water, from natural such as a lake or pond to artificial like a moat or canal.
Tag 3. Its key is 'type', which means Type of a relation. The tag belongs to a tag group 'properties'. The tag value is multipolygon.
Tag 4. 'water:lake'. The tag belongs to a tag group 'water'. This tag means: A natural or semi-natural body of relatively still fresh or salt water which is surrounded by land.

Caption:
Covering nearly half the frame, West Lake emerges as an expansive, irregular water body, stretching from the top boundary toward the midpoint. Portions extend beyond the view, while trees or landscaped areas could line its perimeter, implying a tranquil setting suitable for recreation.

# Task

Raw:
The selected area locates at the {center_coarse} of the image patch. It has a {shape} shape. It occupies {normed_area} of the image patch. Its smoothed boundary is {geometry}.
{clip}It has following tags:
{tags}

Caption:
```

## Prompt template for Task 2: Non-area Description

```text
You are a highly qualified expert trained to caption remote sensing images. Your task is to produce a single, coherent paragraph of about 50 words describing the chosen non-area feature in the image. You will be given relevant OpenStreetMap (OSM) information. Follow the instructions below to complete the task.

# Instruction

1. **Description Focus**:
   - Describe the feature naturally within the scene, avoiding phrases like “this selected non-area” or “this element.”  
   - Emphasize position (e.g., “running along the upper-left edge”), orientation (e.g., “oriented north-south”), and approximate extent (“spanning most of the image width”).  
   - Mention shape (straight, gently curved, winding) and key visually significant attributes or uses (e.g., a highway, a railroad line).  
   - If the feature has multiple segments, describe their relative positions and shapes within the scene.  
   - Incorporate straightforward interpretations from the provided data and any logical inferences.  
   - Include brief references to surroundings (e.g., buildings, fields) if they can be reasonably inferred. If the data does not confirm their presence, use cautious language (e.g., “probably,” “possibly,” “likely”).


2. **Use of Provided OSM Data**:

    - Convert coordinates or length measurements into qualitative terms (e.g., “extending across much of the view”).  
    - Present confirmed data factually (e.g., if it is a county road, say so).  
    - Do not quote or reveal raw keys or tags.  
    - Refrain from using phrases like “tagged as” or “the tag indicates.”  
    - Employ uncertainty phrases (“seems,” “likely,” “probably”) only when the data does not confirm a detail.

3. **Inferences and Naming Conventions**:
    - Use real-world names from the data if available (e.g., “Old River Road”).  
    - If you must speculate (e.g., about road surface), do so with cautious language: “possibly paved.”
    - For highways or expressways, when only one direction is described in the data but it is reasonable to infer parallel lanes for opposing traffic, include such speculation using phrases like "likely" or "typically."

4. **Coordinate and Orientation Note**:
    - Coordinates range from (0, 0) at the bottom-left to (1, 1) at the top-right. In this system, the left side corresponds to the east and the right side to the west.
    - The square root of the area of the entire Region of Interest (ROI) is normalized to a length of 1.
    - If an endpoint touches the edge of the image patch, an “_edge” is added to its position (e.g., `top-left_edge` or `center-right_edge`).

5. **Fluency**:
    - Maintain fluent, cohesive language.  
    - Aim for around 50 words in a single paragraph.
    - Avoid using underscores or other special characters in the caption.

6. **Output Language Flexibility**:
    - Write exclusively in English, avoiding non-English characters or symbols.
    - Avoid using special characters like underscores in the caption.
    - While examples are provided for guidance, you are encouraged to use varied and natural phrasing that best fits the context, ensuring clarity and simplicity.

# Examples

## Example 1

Raw:
The selected non-area has endpoints: one located at the right-center_right, and the other at the left-top_left. Its sinuosity is described as straight, and its orientation is east-west. It has a normalized length of 1.001 and a real length of 272 meters. The geometry is defined as {{[(1.000, 0.604), (0.000, 0.667)]}}.
Some parts of the geometry extend beyond this ROI.

It has the following tags:
Tag 1. Its key is 'name', which means The primary name: in general, the most prominent signposted name or the most common name in the local language(s). The tag belongs to a tag group 'names'. The tag value is BNSF Glasgow Subdivision.
Tag 2. 'usage:main'. The tag belongs to a tag group ''. This tag means: Specifies that a railway is a main line, presumably heavy traffic, often double tracked and/or electrified, may be high speed.
Tag 3. Its key is 'electrified', which means Indicates infrastructure to supply vehicles with electricity, on rail tracks or highways. The tag belongs to a tag group ''. The tag value is no.
Tag 4. Its key is 'gauge', which means Used to describe the 'gauge' (distance between the inside of the rails) on railways. The tag belongs to a tag group 'railways'. The tag value is 1435.
Tag 5. Its key is 'maxspeed', which means Specifies the maximum legal speed limit on a road, railway or waterway. The tag belongs to a tag group 'restrictions'. The tag value is 79 mph.
Tag 6. Its key is 'maxspeed:freight' and value is '70 mph'. The tag can be interpreted by its key and value
Tag 7. Its key is 'old_railway_operator', which means Used for indicating the former operator of a railway line, whether active or abandoned. The tag belongs to a tag group ''. The tag value is GN.
Tag 8. Its key is 'operator', which means Сompany, corporation, person or any other entity who is directly in charge of the current operation of a map object. The tag belongs to a tag group 'properties'. The tag value is BNSF Railway.
Tag 9. 'railway:rail'. The tag belongs to a tag group 'railways'. This tag means: Rails of a typical gauge track.
Tag 10. Its key is 'railway:track_ref', which means The reference number of a track within a station, yard or other railway facility. The tag belongs to a tag group 'railways'. The tag value is 2.


Caption:
The BNSF Glasgow Subdivision railway, a mainline track operated by BNSF Railway, runs east-west across the scene, spanning nearly the full width. This straight railway, featuring a standard gauge of 1435 mm, is not electrified. With a maximum speed limit of 79 mph, it supports heavy freight traffic, surrounded by open fields and scattered structures.

## Example 2

Raw:
The selected non-area has endpoints: one located at the right-bottom_right, and the other at the left-bottom_left. Its sinuosity is described as broken, and its orientation is too curved or twisted to determine accurately. It has a normalized length of 1.000 and a real length of 268 meters. The geometry is defined as {{[(1.000, 0.009), (-0.000, 0.046)]}}.
Some parts of the geometry extend beyond this ROI.

It has the following tags:
Tag 1. 'highway:residential'. The tag belongs to a tag group 'highways'. This tag means: Road in a residential area
Tag 2. Its key is 'name', which means The primary name: in general, the most prominent signposted name or the most common name in the local language(s). The tag belongs to a tag group 'names'. The tag value is Redbud Lane.
Tag 3. Its key is 'tiger:county', which means the county and state where a feature is located, formatted as "County Name, State Abbreviation.". The tag belongs to a tag group 'customized tag'. The tag value is Miller, MO.

Caption:
A winding residential road named Redbud Lane crosses the lower part of the image, extending from the bottom-right edge to the bottom-left edge. The road’s serpentine shape meanders gently through a residential area, probably bordered by small clusters of houses and landscaped yards. Its approximate length spans the width of the view, connecting neighborhoods in Miller County, Missouri.

## Example 3

Raw:
The selected non-area has endpoints: one located at the right-bottom_right, and the other at the left-top_top. Its sinuosity is described as curved, and its orientation is northwest-southeast. It has a normalized length of 1.258 and a real length of 340 meters. The geometry is defined as {{[(1.000, 0.187), (0.056, 1.000)]}}.
Some parts of the geometry extend beyond this ROI.

It has the following tags:
Tag 1. 'highway:path'. The tag belongs to a tag group 'highways'. This tag means: A generic path used by pedestrians, small vehicles, for animal riding or livestock walking. Not used by two-track vehicles. Very broad, non-specific meaning.
Tag 2. Its key is 'name', which means The primary name: in general, the most prominent signposted name or the most common name in the local language(s). The tag belongs to a tag group 'names'. The tag value is Rabb Park Trail #747.

Caption:

A gently curved path, known as "Rabb Park Trail #747," is oriented along a northwest-southeast axis across the image. It spans a significant portion of the view, stretching diagonally from the bottom-right edge to the top-left edge. The trail is likely intended for pedestrian or small vehicle use, potentially accommodating activities such as walking or animal riding.

## Example 4

Raw:
The selected non-area has endpoints: one located at the left-center_left, and the other at the top-center_top. Its sinuosity is described as straight, and its orientation is southwest-northeast. It has a normalized length of 0.636 and a real length of 171 meters. The geometry is defined as {{[(0.000, 0.531), (0.430, 1.000)]}}.
Some parts of the geometry extend beyond this ROI.

It has the following tags:
Tag 1. Its key is 'wires', which means Number of wires per power cable. "single" (1), "double" (2), "triple" (3) or "quad" (4). The tag belongs to a tag group 'power'. The tag value is double.
Tag 2. Its key is 'voltage', which means For describing the voltage of power lines, cables, and substations. The tag belongs to a tag group 'power'. The tag value is 345000.
Tag 3. Its key is 'operator', which means Сompany, corporation, person or any other entity who is directly in charge of the current operation of a map object. The tag belongs to a tag group 'properties'. The tag value is PNM.
Tag 4. 'power:line'. The tag belongs to a tag group 'power'. This tag means: High-voltage power lines used for power transmission, usually supported by towers or pylons
Tag 5. Its key is 'cables', which means Number of electrically separated, individual or bundled, power-carrying conductors in a power line or cable. The tag belongs to a tag group 'power'. The tag value is 3.
Tag 6. Its key is 'frequency', which means For describing the frequency of electric power supply networks and devices, electrified railways or buses, and communications devices. The unit is Hz (cycles per second). The tag belongs to a tag group 'properties'. The tag value is 60.

Caption:
Stretching from the left-center edge to the top-center edge of the scene, a high-voltage power line is aligned along a southwest-northeast axis. Measuring 171 meters, it is supported by pylons likely beyond the visible area. The line features three cables carrying 345,000 volts, operated by PNM, and transmits power at 60 Hz with double wires per cable.

## Example 5

Raw:
The selected non-area has endpoints: one located at the top-center_top, and the other at the right-top_top. Its sinuosity is described as twisted, and its orientation is too curved or twisted to determine accurately. It has a normalized length of 0.970 and a real length of 261 meters. The geometry is defined as {{[(-0.000, 0.863), (0.077, 1.000)], [(0.121, 1.000), (0.414, 1.000)], [(0.600, 1.000), (0.737, 1.000)]}}.
Some parts of the geometry extend beyond this ROI.

It has the following tags:
Tag 1. 'bicycle:designated'. The tag belongs to a tag group ''. This tag means: Roads and other objects designated/signed to use for cyclists
Tag 2. 'foot:designated'. The tag belongs to a tag group ''. This tag means: A preferred or designated route for pedestrians.
Tag 3. 'highway:path'. The tag belongs to a tag group 'highways'. This tag means: A generic path used by pedestrians, small vehicles, for animal riding or livestock walking. Not used by two-track vehicles. Very broad, non-specific meaning.
Tag 4. Its key is 'horse', which means Access permission for equestrians. The tag belongs to a tag group 'restrictions'. The tag value is designated.
Tag 5. 'motor_vehicle:forestry'. The tag belongs to a tag group 'restrictions'. This tag means: The use of the road is authorized exclusively for all types of motor vehicle which use it for forestry purposes
Tag 6. Its key is 'name', which means The primary name: in general, the most prominent signposted name or the most common name in the local language(s). The tag belongs to a tag group 'names'. The tag value is Auger Hole Trail.
Tag 7. Its key is 'sac_scale', which means A difficulty rating scheme for hiking trails. The tag belongs to a tag group 'properties'. The tag value is mountain_hiking.
Tag 8. Its key is 'smoothness', which means Specifies the physical usability of a way for wheeled vehicles due to surface regularity/flatness. The tag belongs to a tag group 'properties'. The tag value is very_bad.
Tag 9. 'surface:ground'. The tag belongs to a tag group ''. This tag means: No special surface, the ground itself has marks of human or animal usage. This value gives only a rough description; if possible, use a more precise value such as grass, clay, sand, earth, gravel or pebblestone.
Tag 10. Its key is 'symbol', which means Human-Readable Description of Route Symbols that are used as waymarkers or on guideposts. The tag belongs to a tag group 'properties'. The tag value is orange squares.
Tag 11. Its key is 'trail_visibility', which means Classification scheme for trail (not route) visibility and way-finding skill required. The tag belongs to a tag group 'properties'. The tag value is excellent.

Caption:
Auger Hole Trail, a winding path with orange square markers, extends across the northern edge of the scene. Likely surrounded by rugged terrain, it suits mountain hiking, with excellent visibility and a rough ground surface. Designed for pedestrians, cyclists, and equestrians, it restricts motor vehicles, except for forestry purposes.

# Task

Raw:
The selected non-area has endpoints: {endpoints_coarse}. Its sinuosity is described as {sinuosity}, and its orientation is {orientation}. It has a normalized length of {normed_length} and a real length of {length} meters. The geometry is defined as {geometry}.
{clip}
It has the following tags:
{tags}

Caption:
```

## Prompt template for Task 3: Caption Revision

```text
You are a highly skilled language model specializing in augmenting remote sensing datasets. Your task is to revise a description (caption) of a remote sensing image while preserving its overall meaning, while introducing variations in tone, length, and phrasing. You may also omit certain details, especially if they appear less confident or speculative in the original caption. Use the provided instructions and examples below to guide your revisions.

# Instructions

1. **Preserve Overall Meaning**:
    - Ensure the revised caption reflects the main idea of the raw caption while allowing for some information to be omitted, particularly if it appears speculative or less confident in the original.
    - Avoid adding new information or altering essential spatial or contextual details (e.g., orientation, position, or scene dimensions).

2. **Introduce Variations**:
    - Use natural, fluent, and cohesive language for the revised captions.
    - Adjust tone (e.g., formal, descriptive, or neutral) and sentence structure.
    - Ensure the revised caption has a length between 10 to 50 words, maintaining clarity and readability.
    - Incorporate diverse sentence structures, including "There be" clauses, descriptive phrases, retrieval-like language, and alternative constructions, to enrich the captions.

3. **Contextual and Positional References**:
    - Spatial References and Orientation:
        + The coordinate system is normalized with (0, 0) at the bottom-left and (1, 1) at the top-right.
        + The left corresponds to the east, the right to the west, the top to the north, and the bottom to the south.
    - Normalized Measurements:
        + For area elements: The total area of the Region of Interest (ROI) is normalized to 1.
        + For non-area elements: The square root of the total ROI area is normalized to a length of 1.

4. **Simplifications for Dominant Elements**:
    - You may simplify the caption only if the described element spans most of the image patch. In such cases, provide a concise, high-level description of the scene that highlights the dominant feature. For instance, you might describe it as "a remote sensing image," "a satellite photo," or "an aerial view" followed by a general description of the prominent feature or land use class.
    - Ensure the simplified caption remains natural and clearly conveys the essence of the dominant element.

5. **Supplementing with Extra Knowledge for Well-Known Locations**:
   - If the scene includes a well-known location (e.g., a famous landmark, nature reserve, or historical site) that you are confident about based on prior knowledge, you may supplement the caption with relevant details.
   - Only include additional information when you are certain it is accurate and relevant to the description of the scene.
   - Integrate this knowledge seamlessly into the caption, ensuring that it enhances the description without overwhelming it or introducing unnecessary detail. The additional information should help paint a clearer picture of the landscape, environment, or features that are visible in the image.

6. **Output Language Flexibility**:
    - Write exclusively in English, avoiding non-English characters or symbols.
    - Avoid using special characters like underscores in the caption.
    - While examples are provided for guidance, you are encouraged to use varied and natural phrasing that best fits the context, ensuring clarity and simplicity.

7. **Avoid Formatting and Labels**:
    - Do not include prefixes like "Revised:" or any labels.
    - Do not use markdown formatting such as bold (**), italics (*), or any other special characters.
    - Output only the revised caption as plain text.

8. **Single Revision Rule**:
    - Generate exactly one revision per input caption. Do not provide multiple variations or versions.

# Examples

## Example 1
Raw: {example1_raw}
Revised: {example1_revised}

## Example 2
Raw: {example2_raw}
Revised: {example2_revised}

## Example 3
Raw: {example3_raw}
Revised: {example3_revised}

## Example 4
Raw: {example4_raw}
Revised: {example4_revised}

## Example 5
Raw: {example5_raw}
Revised: {example5_revised}

# Task
Raw: {raw}
Revised:
```
