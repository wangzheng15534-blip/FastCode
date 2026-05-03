#!/usr/bin/env node
"use strict";

const path = require("path");
let ts;
try {
  ts = require("typescript");
} catch (error) {
  console.error("typescript module is required for semantic extraction");
  process.exit(2);
}

const files = process.argv.slice(2).map((file) => path.resolve(file));
const options = {
  allowJs: true,
  checkJs: false,
  noEmit: true,
  target: ts.ScriptTarget.ES2020,
  module: ts.ModuleKind.CommonJS,
  moduleResolution: ts.ModuleResolutionKind.NodeJs,
  jsx: ts.JsxEmit.Preserve,
  skipLibCheck: true,
};
const program = ts.createProgram(files, options);
const checker = program.getTypeChecker();
const fileSet = new Set(files.map((file) => path.normalize(file)));
const facts = {
  imports: [],
  calls: [],
  inherits: [],
  stats: {
    files: files.length,
    imports: 0,
    calls: 0,
    inherits: 0,
    unresolved_calls: 0,
  },
};

function rel(fileName) {
  return path.relative(process.cwd(), fileName).replace(/\\/g, "/");
}

function pos(source, node) {
  const p = source.getLineAndCharacterOfPosition(node.getStart(source));
  return { line: p.line + 1, col: p.character };
}

function declarationInfo(symbol) {
  if (!symbol) return null;
  const declarations = symbol.getDeclarations() || [];
  for (const declaration of declarations) {
    const source = declaration.getSourceFile();
    if (!source || source.isDeclarationFile) continue;
    if (!fileSet.has(path.normalize(source.fileName))) continue;
    const p = pos(source, declaration);
    return {
      path: rel(source.fileName),
      line: p.line,
      col: p.col,
      name: symbol.getName(),
      symbol: checker.getFullyQualifiedName(symbol).replace(/^"|"$/g, ""),
    };
  }
  return null;
}

function importTarget(moduleText, sourceFile) {
  const resolved = ts.resolveModuleName(moduleText, sourceFile.fileName, options, ts.sys);
  const fileName = resolved && resolved.resolvedModule && resolved.resolvedModule.resolvedFileName;
  if (!fileName) return null;
  if (!fileSet.has(path.normalize(fileName))) return null;
  return rel(fileName);
}

function visit(sourceFile, node) {
  if (ts.isImportDeclaration(node) && node.moduleSpecifier && ts.isStringLiteral(node.moduleSpecifier)) {
    const moduleText = node.moduleSpecifier.text;
    const targetPath = importTarget(moduleText, sourceFile);
    if (targetPath) {
      const p = pos(sourceFile, node);
      facts.imports.push({
        source_path: rel(sourceFile.fileName),
        target_path: targetPath,
        module: moduleText,
        source_line: p.line,
        source_col: p.col,
      });
    }
  }

  if (ts.isCallExpression(node)) {
    const signature = checker.getResolvedSignature(node);
    const declaration = signature && signature.getDeclaration && signature.getDeclaration();
    const symbol = declaration && checker.getSymbolAtLocation(declaration.name || declaration);
    const info = declarationInfo(symbol);
    if (info) {
      const p = pos(sourceFile, node.expression);
      facts.calls.push({
        source_path: rel(sourceFile.fileName),
        target_path: info.path,
        call_name: node.expression.getText(sourceFile),
        target_name: info.name,
        target_symbol: info.symbol,
        source_line: p.line,
        source_col: p.col,
        target_line: info.line,
        target_col: info.col,
      });
    } else {
      facts.stats.unresolved_calls += 1;
    }
  }

  if (
    (ts.isClassDeclaration(node) || ts.isInterfaceDeclaration(node)) &&
    node.name &&
    node.heritageClauses
  ) {
    const sourcePos = pos(sourceFile, node.name);
    for (const clause of node.heritageClauses) {
      for (const heritageType of clause.types || []) {
        const heritageSymbol = checker.getSymbolAtLocation(heritageType.expression);
        const info = declarationInfo(heritageSymbol);
        if (!info) continue;
        facts.inherits.push({
          source_path: rel(sourceFile.fileName),
          source_name: node.name.text,
          source_line: sourcePos.line,
          source_col: sourcePos.col,
          target_path: info.path,
          target_name: info.name,
          target_symbol: info.symbol,
          target_line: info.line,
          target_col: info.col,
          relation_kind:
            clause.token === ts.SyntaxKind.ImplementsKeyword
              ? "implements"
              : "extends",
        });
      }
    }
  }

  ts.forEachChild(node, (child) => visit(sourceFile, child));
}

for (const sourceFile of program.getSourceFiles()) {
  if (sourceFile.isDeclarationFile) continue;
  if (!fileSet.has(path.normalize(sourceFile.fileName))) continue;
  visit(sourceFile, sourceFile);
}

facts.stats.imports = facts.imports.length;
facts.stats.calls = facts.calls.length;
facts.stats.inherits = facts.inherits.length;
process.stdout.write(JSON.stringify(facts));
