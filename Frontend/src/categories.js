// Keep thresholds in sync with categorize() in Backend/app/llm.py
export const CATEGORIES = [
  { name: "Highly Suitable", className: "cat-high", above: 77, range: "Score 78–100" },
  { name: "Suitable", className: "cat-suitable", above: 50, range: "Score 51–77" },
  { name: "Moderate", className: "cat-moderate", above: 40, range: "Score 41–50" },
  { name: "Not Suitable", className: "cat-not", above: -1, range: "40 or under, or missing a must-have" },
];

// Derived from the score so results already on screen follow the current thresholds
export function categorize(result) {
  if (result.status !== "ok") return result;
  const category = result.failed_must_haves?.length
    ? "Not Suitable"
    : CATEGORIES.find((c) => result.score > c.above).name;
  return { ...result, category };
}

export const CRITERIA = [
  { key: "skills", label: "Skills" },
  { key: "experience", label: "Experience" },
  { key: "education", label: "Education & certifications" },
  { key: "fit", label: "Role fit" },
];

export function categoryClass(name) {
  return CATEGORIES.find((c) => c.name === name)?.className || "cat-error";
}

// Lower is better; errors sort last
export function categoryRank(result) {
  if (result.status !== "ok") return CATEGORIES.length;
  return CATEGORIES.findIndex((c) => c.name === result.category);
}
