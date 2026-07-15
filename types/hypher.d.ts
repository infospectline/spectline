declare module "hypher" {
  export default class Hypher {
    constructor(patterns: any);
    hyphenate(word: string): string[];
    hyphenateText?(text: string, splitter?: string): string;
  }
}