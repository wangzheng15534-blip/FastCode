package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type ImportFact struct {
	SourcePath string `json:"source_path"`
	TargetPath string `json:"target_path,omitempty"`
	ImportPath string `json:"import_path"`
	SourceLine int    `json:"source_line"`
	SourceCol  int    `json:"source_col"`
}

type CallFact struct {
	SourcePath   string `json:"source_path"`
	TargetPath   string `json:"target_path"`
	CallName     string `json:"call_name"`
	TargetName   string `json:"target_name"`
	TargetSymbol string `json:"target_symbol"`
	SourceLine   int    `json:"source_line"`
	SourceCol    int    `json:"source_col"`
	TargetLine   int    `json:"target_line"`
	TargetCol    int    `json:"target_col"`
}

type InheritFact struct {
	SourcePath string `json:"source_path"`
	SourceName string `json:"source_name"`
	SourceLine int    `json:"source_line"`
	SourceCol  int    `json:"source_col"`
	TargetPath string `json:"target_path"`
	TargetName string `json:"target_name"`
	TargetLine int    `json:"target_line"`
	TargetCol  int    `json:"target_col"`
}

type Output struct {
	Imports  []ImportFact   `json:"imports"`
	Calls    []CallFact     `json:"calls"`
	Inherits []InheritFact  `json:"inherits"`
	Stats    map[string]int `json:"stats"`
	Errors   []string       `json:"errors,omitempty"`
}

func rel(path string) string {
	cwd, err := os.Getwd()
	if err != nil {
		return filepath.ToSlash(path)
	}
	out, err := filepath.Rel(cwd, path)
	if err != nil {
		return filepath.ToSlash(path)
	}
	return filepath.ToSlash(out)
}

func unquoteImport(value string) string {
	unquoted, err := strconv.Unquote(value)
	if err != nil {
		return strings.Trim(value, "\"")
	}
	return unquoted
}

func exprName(expr ast.Expr) string {
	switch x := expr.(type) {
	case *ast.Ident:
		return x.Name
	case *ast.SelectorExpr:
		return exprName(x.X) + "." + x.Sel.Name
	default:
		return ""
	}
}

func main() {
	flag.Parse()
	args := flag.Args()
	if len(args) > 0 && args[0] == "--" {
		args = args[1:]
	}
	fset := token.NewFileSet()
	files := make([]*ast.File, 0, len(args))
	fileByName := map[string]*ast.File{}
	fileSet := map[string]bool{}
	for _, arg := range args {
		abs, err := filepath.Abs(arg)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		file, err := parser.ParseFile(fset, abs, nil, 0)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		files = append(files, file)
		fileByName[abs] = file
		fileSet[abs] = true
	}
	info := &types.Info{
		Types: make(map[ast.Expr]types.TypeAndValue),
		Defs:  make(map[*ast.Ident]types.Object),
		Uses:  make(map[*ast.Ident]types.Object),
	}
	conf := types.Config{Importer: importer.Default(), Error: func(error) {}}
	_, _ = conf.Check("fastcode_semantic", fset, files, info)

	out := Output{Stats: map[string]int{"files": len(files)}}
	for abs, file := range fileByName {
		sourcePath := rel(abs)
		for _, imp := range file.Imports {
			pos := fset.Position(imp.Pos())
			out.Imports = append(out.Imports, ImportFact{
				SourcePath: sourcePath,
				ImportPath: unquoteImport(imp.Path.Value),
				SourceLine: pos.Line,
				SourceCol:  pos.Column - 1,
			})
		}
		ast.Inspect(file, func(node ast.Node) bool {
			if gen, ok := node.(*ast.GenDecl); ok && gen.Tok == token.TYPE {
				for _, spec := range gen.Specs {
					typeSpec, ok := spec.(*ast.TypeSpec)
					if !ok {
						continue
					}
					sourcePos := fset.Position(typeSpec.Name.Pos())
					switch typed := typeSpec.Type.(type) {
					case *ast.StructType:
						for _, field := range typed.Fields.List {
							if len(field.Names) != 0 {
								continue
							}
							targetName := exprName(field.Type)
							if targetName == "" {
								continue
							}
							var obj types.Object
							switch expr := field.Type.(type) {
							case *ast.Ident:
								obj = info.Uses[expr]
							case *ast.SelectorExpr:
								obj = info.Uses[expr.Sel]
							}
							if obj == nil {
								continue
							}
							objPos := fset.Position(obj.Pos())
							if objPos.Filename == "" {
								continue
							}
							targetAbs, err := filepath.Abs(objPos.Filename)
							if err != nil || !fileSet[targetAbs] {
								continue
							}
							out.Inherits = append(out.Inherits, InheritFact{
								SourcePath: sourcePath,
								SourceName: typeSpec.Name.Name,
								SourceLine: sourcePos.Line,
								SourceCol:  sourcePos.Column - 1,
								TargetPath: rel(targetAbs),
								TargetName: obj.Name(),
								TargetLine: objPos.Line,
								TargetCol:  objPos.Column - 1,
							})
						}
					case *ast.InterfaceType:
						for _, field := range typed.Methods.List {
							if len(field.Names) != 0 {
								continue
							}
							targetName := exprName(field.Type)
							if targetName == "" {
								continue
							}
							var obj types.Object
							switch expr := field.Type.(type) {
							case *ast.Ident:
								obj = info.Uses[expr]
							case *ast.SelectorExpr:
								obj = info.Uses[expr.Sel]
							}
							if obj == nil {
								continue
							}
							objPos := fset.Position(obj.Pos())
							if objPos.Filename == "" {
								continue
							}
							targetAbs, err := filepath.Abs(objPos.Filename)
							if err != nil || !fileSet[targetAbs] {
								continue
							}
							out.Inherits = append(out.Inherits, InheritFact{
								SourcePath: sourcePath,
								SourceName: typeSpec.Name.Name,
								SourceLine: sourcePos.Line,
								SourceCol:  sourcePos.Column - 1,
								TargetPath: rel(targetAbs),
								TargetName: obj.Name(),
								TargetLine: objPos.Line,
								TargetCol:  objPos.Column - 1,
							})
						}
					}
				}
			}
			call, ok := node.(*ast.CallExpr)
			if !ok {
				return true
			}
			name := exprName(call.Fun)
			if name == "" {
				return true
			}
			var obj types.Object
			switch fun := call.Fun.(type) {
			case *ast.Ident:
				obj = info.Uses[fun]
			case *ast.SelectorExpr:
				obj = info.Uses[fun.Sel]
			}
			if obj == nil {
				out.Stats["unresolved_calls"]++
				return true
			}
			objPos := fset.Position(obj.Pos())
			if objPos.Filename == "" {
				out.Stats["unresolved_calls"]++
				return true
			}
			targetAbs, err := filepath.Abs(objPos.Filename)
			if err != nil || !fileSet[targetAbs] {
				out.Stats["unresolved_calls"]++
				return true
			}
			pos := fset.Position(call.Fun.Pos())
			out.Calls = append(out.Calls, CallFact{
				SourcePath:   sourcePath,
				TargetPath:   rel(targetAbs),
				CallName:     name,
				TargetName:   obj.Name(),
				TargetSymbol: obj.Pkg().Path() + "." + obj.Name(),
				SourceLine:   pos.Line,
				SourceCol:    pos.Column - 1,
				TargetLine:   objPos.Line,
				TargetCol:    objPos.Column - 1,
			})
			return true
		})
	}
	out.Stats["imports"] = len(out.Imports)
	out.Stats["calls"] = len(out.Calls)
	out.Stats["inherits"] = len(out.Inherits)
	_ = json.NewEncoder(os.Stdout).Encode(out)
}
