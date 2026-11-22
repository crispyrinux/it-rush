import wordCatalog from "./bisindoWords.json";

export const ONE_HAND = ["C", "E", "I", "J", "L", "O", "R", "U", "V", "Z"];

export const UNITS = [
  { id: 1, title: "Unit 1: Vokal", color: "bg-sun", letters: ["A", "E", "I", "O", "U"] },
  { id: 2, title: "Unit 2: Isyarat 1 Tangan", color: "bg-mint", letters: ["C", "J", "L", "R", "V", "Z"] },
  { id: 3, title: "Unit 3: Isyarat 2 Tangan Dasar", color: "bg-coral", letters: ["B", "D", "F", "H", "K", "T", "W", "X"] },
  { id: 4, title: "Unit 4: Isyarat 2 Tangan Lanjutan", color: "bg-sky", letters: ["G", "M", "N", "P", "Q", "S", "Y"] },
];

export const handsOf = (l: string) => (ONE_HAND.includes(l) ? 1 : 2);

export interface WordItem {
  id: string;
  title: string;
  hands: 1 | 2;
  guide: string;
  localVideoPath?: string;
}

export const WORD_MODEL_VERSION: string = wordCatalog.modelVersion;

export const WORD_ITEMS: WordItem[] = wordCatalog.words.map((item) => ({
  id: item.id,
  title: item.title,
  hands: item.hands === 1 ? 1 : 2,
  guide: item.guide,
  localVideoPath: item.localVideoPath,
}));
