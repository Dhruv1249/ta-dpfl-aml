#!/usr/bin/env python3
"""Consolidate completed TA-DPFL DP-run evidence without retraining or test tuning.

Reads only artifacts already produced by the run and optional validation / gradient
sanity diagnostics. Outputs JSON + Markdown summary. It does not load or change
model checkpoints, run training, select thresholds, or alter DP accounting.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists(): return None
    with path.open('r', encoding='utf-8') as f: return json.load(f)


def fnum(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError): return None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--run-dir', default=None)
    ap.add_argument('--preflight', default=None)
    ap.add_argument('--output-dir', default=None)
    args=ap.parse_args()
    cfg_path=Path(args.config).resolve()
    with cfg_path.open('r',encoding='utf-8') as f: cfg=yaml.safe_load(f)
    run_dir=Path(args.run_dir).resolve() if args.run_dir else Path(cfg['output_dir'])
    if not run_dir.is_absolute(): run_dir=(Path.cwd()/run_dir).resolve()
    output_dir=Path(args.output_dir).resolve() if args.output_dir else run_dir/'diagnostics'
    output_dir.mkdir(parents=True,exist_ok=True)

    overall_path=run_dir/'overall_results.csv'
    typology_path=run_dir/'typology_results.csv'
    train_path=run_dir/'training_log.csv'
    overall=pd.read_csv(overall_path) if overall_path.exists() else pd.DataFrame()
    typology=pd.read_csv(typology_path) if typology_path.exists() else pd.DataFrame()
    train=pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()

    expected_rounds=int(cfg.get('federated',{}).get('rounds',cfg.get('experiments',{}).get('rounds',3200)))
    expected_clients=int(cfg.get('federated',{}).get('clients_per_round',4) or 4)
    privacy_cfg=cfg.get('privacy',{})
    preflight=read_json(Path(args.preflight).resolve()) if args.preflight else None
    assumptions=read_json(run_dir/'partial_dp_accounting_assumptions.json')
    comparison=read_json(run_dir/'privacy_comparison.json')
    schedule=read_json(run_dir/'schedule_config.json')
    exposure=read_json(run_dir/'partial_dp_client_exposure_counts.json')

    summary={'run_dir':str(run_dir),'config':str(cfg_path),'checks':{},'metrics':{},'privacy':{},'diagnostics':{},'interpretation':{},'missing_artifacts':[]}
    for p in [overall_path,typology_path,train_path,run_dir/'privacy_comparison.json',run_dir/'partial_dp_accounting_assumptions.json']:
        if not p.exists(): summary['missing_artifacts'].append(str(p))

    # Overall test metric summary. AP is compared to prevalence but no model is selected here.
    if not overall.empty:
        method_rows=[]
        for _,r in overall.iterrows():
            prevalence=fnum(r.get('prevalence'))
            apv=fnum(r.get('AP'))
            auc=fnum(r.get('ROC_AUC'))
            method_rows.append({
                'method':str(r.get('method')),'seed':int(r.get('seed')) if fnum(r.get('seed')) is not None else None,
                'n':int(r.get('n')) if fnum(r.get('n')) is not None else None,
                'positives':int(r.get('positives')) if fnum(r.get('positives')) is not None else None,
                'prevalence':prevalence,'AP':apv,'AP_over_prevalence':(apv/prevalence if apv is not None and prevalence else None),
                'ROC_AUC':auc,'diagnostic_inverted_AUC':(1.0-auc if auc is not None else None),
                'precision':fnum(r.get('precision')),'recall':fnum(r.get('recall')),
                'F1':fnum(r.get('F1')),'false_positive_rate':fnum(r.get('false_positive_rate')),
                'global_epsilon':fnum(r.get('global_epsilon')),'delta':fnum(r.get('delta'))
            })
        summary['metrics']['overall_test']=method_rows
        summary['checks']['all_overall_ap_at_or_above_prevalence']=all(
            x['AP'] is not None and x['prevalence'] is not None and x['AP']>=x['prevalence'] for x in method_rows)
        summary['checks']['all_raw_auc_at_or_above_half']=all(x['ROC_AUC'] is not None and x['ROC_AUC']>=0.5 for x in method_rows)
        summary['checks']['single_seed_only']=len(set(x['seed'] for x in method_rows if x['seed'] is not None))<=1
    else:
        summary['checks']['overall_results_present']=False

    # Typology AP / AUC: capture only aggregate extrema and the per-row values.
    if not typology.empty and {'typology','method','AP','ROC_AUC','prevalence'}.issubset(typology.columns):
        summary['metrics']['typology_rows']=typology[[c for c in ['typology','method','n','positives','prevalence','AP','ROC_AUC'] if c in typology.columns]].replace({np.nan:None}).to_dict('records')
        summary['checks']['typology_rows_all_auc_below_half']=bool((pd.to_numeric(typology['ROC_AUC'],errors='coerce')<0.5).all())
        summary['checks']['typology_rows_all_ap_below_stratum_prevalence']=bool((pd.to_numeric(typology['AP'],errors='coerce')<pd.to_numeric(typology['prevalence'],errors='coerce')).all())

    # Training completeness and optimizer verification.
    if not train.empty and {'method','round','clients','server_optimizer'}.issubset(train.columns):
        per=[]
        for method,g in train.groupby('method'):
            rounds_seen=int(g['round'].nunique())
            server_opts=sorted(str(x) for x in g['server_optimizer'].dropna().unique())
            clients_values=sorted(int(x) for x in pd.to_numeric(g['clients'],errors='coerce').dropna().unique())
            per.append({'method':str(method),'rows':int(len(g)),'unique_rounds':rounds_seen,'min_round':int(g['round'].min()),'max_round':int(g['round'].max()),'server_optimizer':server_opts,'clients_per_round_values':clients_values,'mean_aggregated_delta_l2':fnum(pd.to_numeric(g.get('aggregated_client_delta_l2'),errors='coerce').mean()) if 'aggregated_client_delta_l2' in g else None,'mean_server_update_l2':fnum(pd.to_numeric(g.get('server_update_l2'),errors='coerce').mean()) if 'server_update_l2' in g else None})
        summary['metrics']['training_by_method']=per
        summary['checks']['all_methods_have_expected_rounds']=bool(per) and all(x['unique_rounds']==expected_rounds and x['rows']==expected_rounds for x in per)
        summary['checks']['all_rounds_have_expected_client_count']=all(x['clients_per_round_values']==[expected_clients] for x in per)
        configured_server = cfg.get('federated',{}).get('dp_server_optimizer',cfg.get('federated',{}).get('server_optimizer'))
        summary['checks']['server_optimizer_matches_config'] = (all(x['server_optimizer']==[str(configured_server).lower()] for x in per) if configured_server else None)
        summary['metrics']['configured_server_optimizer'] = str(configured_server).lower() if configured_server else None
        summary['checks']['raw_private_losses_and_sample_counts_not_logged']=bool(train[['mean_sampled_per_local_step','total_sampled_records','mean_local_loss']].isna().all().all()) if set(['mean_sampled_per_local_step','total_sampled_records','mean_local_loss']).issubset(train.columns) else None

    # Schedule structural checks.
    if schedule:
        focus_counts=schedule.get('global_focus_step_counts',{})
        schedule_list=schedule.get('schedule',[])
        summary['metrics']['focus_step_counts']=focus_counts
        summary['checks']['focus_schedule_length_matches_rounds']=len(schedule_list)==expected_rounds
        summary['checks']['focus_count_sum_matches_rounds']=sum(int(v) for v in focus_counts.values())==expected_rounds
        summary['checks']['focus_schedule_is_public_label_independent']=bool(schedule.get('schedule_is_public_and_label_independent',False))
    if exposure:
        part=exposure.get('client_participation_counts',{})
        focus=exposure.get('client_focus_counts',{})
        expected_appearances=(expected_rounds*expected_clients/len(part)) if part else None
        summary['checks']['all_clients_exactly_expected_appearances']=bool(part) and expected_appearances is not None and float(expected_appearances).is_integer() and all(int(v)==int(expected_appearances) for v in part.values())
        summary['metrics']['virtual_client_count']=len(part)
        summary['metrics']['expected_appearances_per_client']=expected_appearances
        if focus:
            per_group={}
            groups=sorted({k for d in focus.values() for k in d})
            for g in groups:
                vals=[int(d.get(g,0)) for d in focus.values()]
                per_group[g]={'min':min(vals),'median':float(np.median(vals)),'max':max(vals),'clients_zero':sum(v==0 for v in vals)}
            summary['metrics']['client_focus_exposure_by_typology']=per_group

    # Privacy evidence. Keep per-method and joint-release values distinct.
    if comparison:
        summary['privacy']={
            'regime_a_setup_satisfied':comparison.get('regime_a_setup_satisfied'),
            'matched_maximum_privacy_cap_per_method':comparison.get('matched_maximum_privacy_cap_per_method'),
            'per_method_epsilon':comparison.get('global_epsilon_bounds_per_method_across_configured_seeds',comparison.get('global_epsilon_bounds')),
            'joint_release':comparison.get('joint_release_all_enabled_dp_methods_and_seeds'),
            'interpretation':comparison.get('interpretation'),
        }
        summary['checks']['per_method_privacy_caps_matched']=bool(comparison.get('matched_maximum_privacy_cap_per_method',comparison.get('matched_maximum_privacy_cap',False)))
    if assumptions:
        summary['privacy'].update({
            'privacy_unit':assumptions.get('privacy_unit'),'adjacency_owner_assumption':assumptions.get('ownership'),
            'fixed_normalization':assumptions.get('fixed_normalization'),'opacus_rdp_validation':assumptions.get('opacus_rdp_validation'),
            'formal_privacy_claim_permitted':assumptions.get('formal_privacy_claim_permitted'),
            'claim_scope':assumptions.get('claim_scope'),'joint_release_warning':assumptions.get('joint_release_warning')
        })
        summary['checks']['opacus_rdp_crosscheck_passed']=assumptions.get('opacus_rdp_validation',{}).get('status')=='PASS'
        summary['checks']['formal_dp_certification_available']=bool(assumptions.get('formal_privacy_claim_permitted',False))

    if preflight:
        summary['diagnostics']['preflight']={
            'status':preflight.get('status'),'rounds':preflight.get('rounds'),'clients_per_round':preflight.get('clients_per_round'),
            'appearances_per_client':preflight.get('appearances_per_client'),
            'independent_rdp_validation':preflight.get('independent_rdp_validation'),
            'accountant_calibration_status':preflight.get('accountant_calibration',{}).get('calibration_status'),
            'target_satisfied':preflight.get('accountant_calibration',{}).get('target_satisfied'),
            'joint_release':preflight.get('privacy_comparison',{}).get('joint_release_all_enabled_dp_methods_and_seeds')
        }

    # Approximate Gaussian-noise-only reference for aggregate parameter-update norms.
    # For independent client Gaussian vectors averaged uniformly across K clients:
    # E||mean(noise)||_2 is approximated by sigma*C*sqrt(P)/(B0*sqrt(K)); multiply by LR.
    # This is a scale diagnostic, not an exact finite-dimensional expectation nor an SNR proof.
    try:
        import torch
        ckpt_dir=run_dir/'checkpoints'
        ckpts=sorted(ckpt_dir.glob('*.pt')) if ckpt_dir.exists() else []
        param_count=None
        if ckpts:
            try: state=torch.load(ckpts[0],map_location='cpu',weights_only=True)
            except TypeError: state=torch.load(ckpts[0],map_location='cpu')
            if isinstance(state,dict): param_count=int(sum(v.numel() for v in state.values() if torch.is_tensor(v)))
        if param_count and not train.empty:
            lr=float(cfg['federated']['learning_rate'])
            clip=float(privacy_cfg['clip_norm'])
            norm=float(privacy_cfg['normalization'])
            k=int(cfg['federated'].get('clients_per_round',4))
            schedule_counts=(schedule or {}).get('global_focus_step_counts',{})
            noise_rows=[]
            for method in train['method'].dropna().unique():
                method_s=str(method)
                report_name={'Uniform-DP-FL':'uniform_dp','Individualized-DP-SGD-style':'individualized_dp','TA-DPFL-Policy-U':'ta_policy_u','TA-DPFL-Policy-D':'ta_policy_d'}.get(method_s)
                if not report_name: continue
                report=read_json(run_dir/'privacy_reports'/f'{report_name}.json')
                if not report: continue
                if report_name=='uniform_dp':
                    sigma_mean=float(report['sigma'])
                elif report_name=='individualized_dp':
                    smap=report.get('noise_multipliers',{})
                    # This comparator adds independent Gaussian vectors per disjoint stratum;
                    # their variances add, so use sqrt(sum sigma_g^2) as the equivalent scale.
                    sigma_mean=math.sqrt(sum(float(v)**2 for v in smap.values())) if smap else None
                else:
                    smap=report.get('noise_multipliers',{})
                    total=sum(int(v) for v in schedule_counts.values())
                    sigma_mean=(math.sqrt(sum(int(schedule_counts[g])*(float(smap[g])**2) for g in schedule_counts)/total) if total and all(g in smap for g in schedule_counts) else None)
                if sigma_mean is None: continue
                noise_only=lr*clip*math.sqrt(param_count)*sigma_mean/(norm*math.sqrt(k))
                observed=fnum(pd.to_numeric(train.loc[train['method']==method_s,'aggregated_client_delta_l2'],errors='coerce').mean())
                noise_rows.append({'method':method_s,'parameter_count':param_count,'mean_noise_multiplier':sigma_mean,'noise_only_expected_aggregate_update_l2_approx':noise_only,'observed_mean_aggregate_client_delta_l2':observed,'observed_to_noise_only_ratio':(observed/noise_only if observed is not None and noise_only>0 else None),'assumptions':'one local DP step per client per round; independent Gaussian client noises; uniform mean over K clients; reported model checkpoint parameter count; asymptotic L2 RMS approximation'})
            if noise_rows: summary['diagnostics']['noise_floor_scale_check']=noise_rows
    except Exception as exc:
        summary['diagnostics']['noise_floor_scale_check_error']=f'{type(exc).__name__}: {exc}'

    # Read validation/gradient diagnostics if present.
    vd=run_dir/'dp_validation_score_diagnostic.csv'
    if vd.exists():
        df=pd.read_csv(vd).replace({np.nan:None})
        summary['diagnostics']['validation_score_direction']=df.to_dict('records')
    for p in sorted((run_dir/'diagnostics').glob('gradient_audit_*.json')) if (run_dir/'diagnostics').exists() else []:
        j=read_json(p)
        if j: summary['diagnostics'].setdefault('gradient_direction_audits',{})[p.stem]=j

    # Evidence-based interpretation. Do not turn a poor metric into a success claim.
    overall_rows=summary['metrics'].get('overall_test',[])
    if overall_rows:
        aucs=[x['ROC_AUC'] for x in overall_rows if x['ROC_AUC'] is not None]
        lifts=[x['AP_over_prevalence'] for x in overall_rows if x['AP_over_prevalence'] is not None]
        summary['interpretation']['detection']=('Ranking is below chance for at least one method; do not claim successful AML detection.' if any(a<0.5 for a in aucs) else ('AP is at/above prevalence for every method; interpret alongside multi-seed variability.' if summary['checks'].get('all_overall_ap_at_or_above_prevalence') else 'Mixed ranking/AP evidence; avoid a broad performance claim.'))
        summary['interpretation']['typology_aware_utility']=('No demonstrated overall AP lift over prevalence in this run.' if lifts and not any(v>=1.0 for v in lifts) else 'Inspect per-method/per-typology differences; a single seed is descriptive only.')
    summary['interpretation']['privacy']='The RDP numerical cross-check is accountant validation, not independent end-to-end DP certification. Report transaction-level add/remove scope and composition across jointly released models.'
    summary['interpretation']['design_limitations']=[
        '256 hashed virtual clients are a simulation, not 256 independently participating institutions.',
        'Single-seed comparisons are descriptive, not evidence of statistical significance.',
        'Validation/test records and separately released raw dataset audits are outside the training-record DP claim.',
        'Do not invert scores, alter thresholds, or tune on the test set based on these diagnostics.'
    ]

    (output_dir/'one_day_evidence_audit.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    lines=['# TA-DPFL one-day evidence audit','',f'- Run directory: `{run_dir}`',f'- Config: `{cfg_path}`','', '## Execution and integrity checks','']
    for k,v in summary['checks'].items(): lines.append(f'- **{k}**: `{v}`')
    lines += ['', '## Overall test metrics','', '| Method | AP | AP / prevalence | ROC-AUC | Precision | Recall | ε |','|---|---:|---:|---:|---:|---:|---:|']
    for x in overall_rows: lines.append(f"| {x['method']} | {x['AP']:.8g} | {x['AP_over_prevalence']:.4g} | {x['ROC_AUC']:.6f} | {x['precision']:.6g} | {x['recall']:.6g} | {x['global_epsilon']} |")
    nf=summary.get('diagnostics',{}).get('noise_floor_scale_check',[])
    if nf:
        lines += ['', '## Approximate DP-noise-only update scale','', '| Method | Approx. noise-only update L2 | Observed aggregate update L2 | Observed / noise-only |','|---|---:|---:|---:|']
        for x in nf: lines.append(f"| {x['method']} | {x['noise_only_expected_aggregate_update_l2_approx']:.6g} | {x['observed_mean_aggregate_client_delta_l2']:.6g} | {x['observed_to_noise_only_ratio']:.3f} |")
        lines += ['', 'This is a scale comparison under the listed assumptions, not a formal signal-to-noise estimate or proof that all observed updates are noise.']
    lines += ['', '## Privacy interpretation','',summary['interpretation']['privacy'],'', '## Detection interpretation','',summary['interpretation'].get('detection','Not available.'),'', '## Typology-aware utility interpretation','',summary['interpretation'].get('typology_aware_utility','Not available.'),'', '## Limitations','']
    for s in summary['interpretation']['design_limitations']: lines.append(f'- {s}')
    if summary['missing_artifacts']:
        lines += ['', '## Missing artifacts','']
        for p in summary['missing_artifacts']: lines.append(f'- `{p}`')
    lines += ['', 'Machine-generated audit. Review source files and confirm interpretations before submission.']
    (output_dir/'one_day_evidence_audit.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f"Saved: {output_dir/'one_day_evidence_audit.json'}")
    print(f"Saved: {output_dir/'one_day_evidence_audit.md'}")
    print('\nKey checks:')
    for k,v in summary['checks'].items(): print(f'  {k}: {v}')

if __name__=='__main__': main()
