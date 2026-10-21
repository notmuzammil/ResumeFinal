export const CATEGORIES = [
  { name: "Highly Suitable", className: "cat-high", range: "Score 80–100" },
  { name: "Suitable", className: "cat-suitable", range: "Score 60–79" },
  { name: "Moderate", className: "cat-moderate", range: "Score 40–59" },
  { name: "Not Suitable", className: "cat-not", range: "Under 40 or missing a must-have" },
];

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
